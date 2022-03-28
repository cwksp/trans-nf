# Modified from https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py

import os

import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from torchvision import transforms

from datasets import register


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
        [transforms.ToTensor()]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )


@register('pixelnerf_shapenet')
class PixelnerfShapenet(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, root_path, category, split, n_support, n_query, support_lst=None, repeat=1, viewrng=None,
        image_size=(128, 128), world_scale=1.0
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.base_path = os.path.join(root_path, category + "_" + split)
        self.dataset_name = category

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = split
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and split == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False

        self.n_support = n_support
        self.n_query = n_query
        self.support_lst = support_lst
        if support_lst is not None:
            assert len(support_lst) == n_support
        self.repeat = repeat
        self.viewrng = viewrng

    def __len__(self):
        return len(self.intrins) * self.repeat

    def __getitem__(self, index):
        index %= len(self.intrins)
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        if self.viewrng is not None:
            l, r = self.viewrng
            rgb_paths = rgb_paths[l: r]
            pose_paths = pose_paths[l: r]

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        zip_lst = list(zip(rgb_paths, pose_paths))
        if self.support_lst is None:
            zip_lst_inds = np.random.choice(len(zip_lst), self.n_support + self.n_query, replace=False)
        else:
            rest_inds = []
            for i in range(len(zip_lst)):
                if i not in self.support_lst:
                    rest_inds.append(i)
            rest_inds = np.random.choice(rest_inds, self.n_query, replace=False).tolist()
            zip_lst_inds = self.support_lst + rest_inds
        zip_lst = [zip_lst[i] for i in zip_lst_inds]

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip_lst:
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        # focal = torch.tensor(focal, dtype=torch.float32)

        # result = {
        #     "path": dir_path,
        #     "img_id": index,
        #     "focal": focal,
        #     "c": torch.tensor([cx, cy], dtype=torch.float32),
        #     "images": all_imgs,
        #     "masks": all_masks,
        #     "bbox": all_bboxes,
        #     "poses": all_poses,
        # }
        t = self.n_support
        all_poses = all_poses[:, :3, :4]
        return {
            'support_imgs': all_imgs[:t],
            'support_poses': all_poses[:t],
            'query_imgs': all_imgs[t:],
            'query_poses': all_poses[t:],
            'focal': focal,
            'near': self.z_near,
            'far': self.z_far,
        }
