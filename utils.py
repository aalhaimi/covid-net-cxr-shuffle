import numpy as np
import torch
import torchvision
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa

np.random.seed(42)
torch.manual_seed(42)
ia.seed(42)

# preprocessing techniques
class RemoveScanInfo(object):
    def __call__(self, img):
        w,h = img.size
        img = torchvision.transforms.functional.crop(img, int(h/6), 0, h-int(h/6) , w )
        return img

class HistogramNorm(object):
    def __init__(self):
        self.hnorm = iaa.HistogramEqualization()
    def __call__(self, img):
        img = np.array(img)
        img = self.hnorm.augment_image(img)
        img = Image.fromarray(img)
        return img


def load_alldataset(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data = {}
    for i_batch, sample_batched in enumerate(data_loader):
        data = sample_batched
    return data