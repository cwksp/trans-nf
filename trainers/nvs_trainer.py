import torch
import torchvision
import numpy as np
import einops
import wandb

from .base_trainer import BaseTrainer
from trainers import register
from utils import poses_to_rays, volume_rendering, batched_volume_rendering


@register('nvs_trainer')
class NvsTrainer(BaseTrainer):

    def make_datasets(self):
        super().make_datasets()

        def get_vislist(dataset, n_vis=8):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return [dataset[i] for i in ids]

        if hasattr(self, 'train_loader'):
            np.random.seed(0)
            self.vislist_train = get_vislist(self.train_loader.dataset)
        if hasattr(self, 'test_loader'):
            np.random.seed(0)
            self.vislist_test = get_vislist(self.test_loader.dataset)

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        if self.epoch <= round(self.cfg['max_epoch'] * 0.8):
            lr = base_lr
        else:
            lr = base_lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _adaptive_sample_rays(self, rays_o, rays_d, gt, n_sample):
        B = gt.shape[0]
        inds = []
        fg_n_sample = n_sample // 2
        for i in range(B):
            fg = ((gt[i].min(dim=-1).values < 1).nonzero().view(-1)).cpu().numpy()
            if fg_n_sample <= len(fg):
                fg = np.random.choice(fg, fg_n_sample, replace=False)
            else:
                fg = np.concatenate([fg, np.random.choice(fg, fg_n_sample - len(fg), replace=True)], axis=0)
            rd = np.random.choice(rays_o.shape[1], n_sample - fg_n_sample, replace=False)
            inds.append(np.concatenate([fg, rd], axis=0))

        def subselect(x, inds):
            t = torch.empty(B, len(inds[0]), 3, device=x.device)
            for i in range(B):
                t[i] = x[i][inds[i], :]
            return t

        return subselect(rays_o, inds), subselect(rays_d, inds), subselect(gt, inds)

    def _iter_step(self, data, is_train):
        data = {k: v.cuda() for k, v in data.items()}
        query_imgs = data.pop('query_imgs')
        query_poses = data.pop('query_poses')

        hyponet = self.model_ddp(data)

        B = query_imgs.shape[0]
        H, W = query_imgs.shape[-2:]
        rays_o, rays_d = poses_to_rays(query_poses, H, W, data['focal'][0])

        gt = einops.rearrange(query_imgs, 'b n c h w -> b (n h w) c')
        rays_o = einops.rearrange(rays_o, 'b n h w c -> b (n h w) c')
        rays_d = einops.rearrange(rays_d, 'b n h w c -> b (n h w) c')

        n_sample = self.cfg['train_n_rays']
        if is_train and self.epoch <= self.cfg.get('adaptive_sample_epoch', 0):
            rays_o, rays_d, gt = self._adaptive_sample_rays(rays_o, rays_d, gt, n_sample)
        else:
            ray_ids = np.random.choice(rays_o.shape[1], n_sample, replace=False)
            rays_o, rays_d, gt = map(lambda _: _[:, ray_ids, :], [rays_o, rays_d, gt])

        pred = volume_rendering(
            hyponet, rays_o, rays_d,
            near=data['near'][0],
            far=data['far'][0],
            points_per_ray=self.cfg['train_points_per_ray'],
            use_viewdirs=hyponet.use_viewdirs,
            rand=is_train,
        )
        mses = ((pred - gt)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'loss': loss.item(), 'psnr': psnr.item()}

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)

    def _gen_vis_result(self, tag, vislist):
        self.model_ddp.eval()
        res = []
        for _data in vislist:
            data = dict()
            for k, v in _data.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor([v])
                else:
                    v = v.unsqueeze(0)
                data[k] = v.cuda()
            query_imgs = data.pop('query_imgs')
            query_poses = data.pop('query_poses')

            hyponet = self.model_ddp(data)

            B = query_imgs.shape[0]
            H, W = query_imgs.shape[-2:]
            rays_o, rays_d = poses_to_rays(query_poses, H, W, data['focal'][0]) # rays_o / rays_d: b n h w c

            sup_rays_o, sup_rays_d = poses_to_rays(data['support_poses'], H, W, data['focal'][0])
            rays_o = torch.cat([sup_rays_o, rays_o], dim=1)
            rays_d = torch.cat([sup_rays_d, rays_d], dim=1)

            with torch.no_grad():
                pred = batched_volume_rendering(
                    hyponet, rays_o, rays_d,
                    batch_size=self.cfg['render_ray_batch'],
                    near=data['near'][0],
                    far=data['far'][0],
                    points_per_ray=self.cfg['train_points_per_ray'],
                    use_viewdirs=hyponet.use_viewdirs,
                    rand=False,
                )

            res.extend([
                data['support_imgs'][0].cpu(), query_imgs[0].cpu(),
                einops.rearrange(pred[0], 'n h w c -> n c h w').clamp(0, 1).detach().cpu(),
            ])
            n_support = data['support_imgs'].shape[1]
            n_query = query_imgs.shape[1]

        res = [torch.ones(n_support, 3, H, W), torch.zeros(n_query, 3, H, W)] + res
        res = torch.cat(res, dim=0).detach()
        imggrid = torchvision.utils.make_grid(res, nrow=n_support + n_query)

        if self.enable_tb:
            self.writer.add_image(tag, imggrid, self.epoch)
        if self.enable_wandb:
            wandb.log({tag: wandb.Image(imggrid)}, step=self.epoch)

    def visualize_epoch(self):
        if hasattr(self, 'vislist_train'):
            self._gen_vis_result('vis_train_dataset', self.vislist_train)
        if hasattr(self, 'vislist_test'):
            self._gen_vis_result('vis_test_dataset', self.vislist_test)


@register('nvs_ae_trainer')
class NvsAeTrainer(NvsTrainer):

    def _iter_step(self, data, is_train):
        data['query_imgs'] = data['support_imgs']
        data['query_poses'] = data['support_poses']
        return super()._iter_step(data, is_train)
