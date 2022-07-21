import numpy as np
import torch
from torch import nn
from ltir.utils.func import per_class_iu, fast_hist
from ltir.dataset.cityscapes_dataset_label import cityscapesDataSetLabel
from ltir.model.deeplab import Res_Deeplab
from PIL import Image
from torch.utils import data
from tqdm import tqdm

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def fast_hist(a, b, n):
    k = (a>=0) & (a<n)
    return np.bincount( n*a[k].astype(int)+b[k], minlength=n**2 ).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / ( hist.sum(1)+hist.sum(0)-np.diag(hist) )


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[ input==mapping[ind][0] ] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def eval_single(cfg, model=None, verbose=True, output_file=None):
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    targetloader = data.DataLoader(cityscapesDataSetLabel(cfg.DATA_TGT_DIRECTORY, cfg.DATA_LIST_TARGET_VAL, crop_size=(1024,512),
                                            mean=IMG_MEAN, set="val"), batch_size=1, num_workers=cfg.NUM_WORKERS,shuffle=False,
                                            pin_memory=True)

    # change the mean for different dataset other than CS
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1,3,1,1))
    mean_img = torch.zeros(1, 1)

    if model is None:
        model = Res_Deeplab(cfg.NUM_CLASSES)
        saved_state_dict = torch.load(cfg.restore_opt1)
        model.load_state_dict(saved_state_dict)
        model.eval()
        model.to(cfg.DEVICE)

    # ------------------------------------------------- #
    # compute scores and save them
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    model.eval()
    test_iter = iter(targetloader)
    with torch.no_grad():
        for index in tqdm(range(len(targetloader))):
            image, label , _, _ = next(test_iter)  # 1. get image
            # create mean image
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B, 1, H, W)  # 2. get mean image
            image = image.clone() - mean_img  # 3, image - mean_img
            image = image.to(cfg.DEVICE)

            # forward
            output = model(image)
            output = nn.functional.softmax(output, dim=1)
            output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
            if verbose and index > 0 and index % 100 == 0:
                print('{:d} / {:d}: {:0.2f}'.format(index, len(targetloader), 100 * np.nanmean(per_class_iu(hist))))
        inters_over_union_classes = per_class_iu(hist)
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if verbose:
            display_stats(cfg, inters_over_union_classes, output_file)
        print('\tCurrent mIoU:', computed_miou)


def display_stats(cfg, inters_over_union_classes, output_file=None):
    name_classes = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "light",
            "sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motocycle",
            "bicycle"]
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class] + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
        if output_file is not None:
            output_file.write(name_classes[ind_class] + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2))+'\n')
    if output_file is not None:
        miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        output_file.write("mIoU: "+str(miou))

