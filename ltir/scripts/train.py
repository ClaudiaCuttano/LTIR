import sys
sys.path.append('/home/ccuttano/LTIR')

from ltir.domain_adaptation.config import cfg, cfg_from_file
import argparse
import os
import os.path as osp
import pprint
from ltir.utils import project_root
from ltir.domain_adaptation.train_UDA import train_UDA
import torch
from ltir.model.deeplab import Res_Deeplab
from ltir.model.deeplabv2_mobileNetv2 import get_deeplabv2_mobileNetv2
from ltir.domain_adaptation.train_UDA_DeepLab_MobileNet import train_UDA_DeepLab_MobileNet

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
    cfg.DATA_LIST_SOURCE = str(project_root) +str(cfg.DATA_LIST_SOURCE)
    print('Using config:')
    pprint.pprint(cfg)

    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.DA_METHOD}'
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    if 'DeepLab_init.pth' in cfg.TRAIN.RESTORE_FROM and cfg.TRAIN.MODEL == 'DeepLab':
        model = Res_Deeplab(cfg.NUM_CLASSES)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)

    if cfg.TRAIN.MODEL == 'DeepLab_MobileNet':
        model = get_deeplabv2_mobileNetv2()
        saved_state_dict_mobile = torch.load("../../pretrained_models/mobilenet_v2-7ebf99e0.pth")
        new_params_mobile = model.model_small.state_dict().copy()
        for i in saved_state_dict_mobile:
            i_parts = i.split('.')
            # i_parts[1]!='18': if truncated version
            if not i_parts[0] == "classifier":
                new_params_mobile[i] = saved_state_dict_mobile[i]
        model.model_small.load_state_dict(new_params_mobile)
        model.model_small.fix_bn()

        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.model_big.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.model_big.load_state_dict(new_params)

    if cfg.TRAIN.MODEL == 'DeepLab':
        train_UDA(model, cfg)
    elif cfg.TRAIN.MODEL == 'DeepLab_MobileNet':
        train_UDA_DeepLab_MobileNet(model, cfg)


if __name__ == '__main__':
    main()

