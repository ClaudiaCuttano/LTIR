import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils import data
import imageio

class SYNDataSet(data.Dataset):

    def __init__(self, root_images, root_labels, list_path, crop_size=(11, 11), ignore_label=255, mean=(128, 128, 128),scale=True,mirror=True, max_iters=None):
        self.root_images = root_images
        self.root_labels = root_labels
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip()[4:] for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []

        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(self.root_images+"/"+name).convert('RGB')

        #label = Image.open(osp.join(self.root, "GT/LABELS/%s" % name)) # label from synthia has to be convert to CityS format
        label = np.asarray(imageio.imread(osp.join(self.root_labels, "GT/LABELS/%s" % name), format='PNG-FI'))[:, :, 0]
        label = Image.fromarray(label)

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name

