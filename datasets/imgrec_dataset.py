from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import register


@register('imgrec_dataset')
class ImgrecDataset(Dataset):

    def __init__(self, imageset, width):
        self.imageset = datasets.make(imageset)
        self.transform = transforms.Compose([
            transforms.Resize(width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 1), ##
        ])

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, idx):
        x = self.transform(self.imageset[idx])
        return {'inp': x, 'gt': x} ##
