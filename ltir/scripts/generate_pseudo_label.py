import sys
sys.path.append('/home/ccuttano/LTIR')
import torch.nn as nn
from ltir.domain_adaptation.config import cfg, cfg_from_file
from PIL import Image
import os
import numpy as np
from ltir.model.deeplab import Res_Deeplab
from ltir.dataset.cityscapes_dataset import cityscapesDataSet
from torch.utils import data
import argparse

from ltir.utils import project_root

IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--cfg', type=str, default=None, help='optional config file', )
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    if not os.path.exists(cfg.SAVE):
        os.makedirs(cfg.SAVE)
    model = Res_Deeplab(num_classes=cfg.NUM_CLASSES)
    model.eval()
    model.cuda()

    target_loader = data.DataLoader(cityscapesDataSet(cfg.DATA_TGT_DIRECTORY, cfg.DATA_LIST_TARGET,
                                      crop_size = (1024, 512), mean = IMG_MEAN, set = "train"), batch_size = 1,
                                      num_workers = cfg.NUM_WORKERS, pin_memory=True)
    predicted_label = np.zeros((len(target_loader), 512, 1024))
    predicted_prob = np.zeros((len(target_loader), 512, 1024))
    image_name = []

    interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

    for index, batch in enumerate(target_loader):
        if index % 100 == 0:
            print('%d processed' % index)
        image, _, name = batch
        output = model(image.cuda())
        output = nn.functional.softmax(output, dim=1)
        output = interp(output).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    print(thres)
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print(thres)
    for index in range(len(target_loader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (cfg.SAVE, name))
