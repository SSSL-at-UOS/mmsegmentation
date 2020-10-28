# import required libraries

import cv2
import sys
import json
import argparse
import numpy as np
import os.path as osp


from pathlib import Path
from glob import glob
from tqdm import tqdm

from mmseg.apis import init_segmentor
from mmseg.core.evaluation import get_palette

from skimage.measure import label, regionprops_table
from skimage.morphology import medial_axis

import matplotlib.pyplot as plt

sys.path.append('..')
from shm_tools.shm_utils import imread, imwrite, inference_segmentor_sliding_window, connect_cracks, remove_small_obj


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# read inference configuration json file
parser = argparse.ArgumentParser()
parser.add_argument("img_folder_name")
parser.add_argument("pixel_len")
parser.add_argument("save_result")
args = parser.parse_args()

lenPerPixel = np.float32(args.pixel_len)
img_folder = args.img_folder_name
save_result = args.save_result

img_path_list = glob(osp.join(img_folder, '*.jpg')) + glob(osp.join(img_folder, '*.jpeg'))
img_path_list = sorted(img_path_list)

config = osp.join('shm_work_dirs', 'deeplabv3plus_r101-d8_769x769_40k_concrete_damage_cs.py')
checkpoint = osp.join('shm_work_dirs', 'iter_40000.pth')

# load the model on GPU
device = 'cuda:0'
model = init_segmentor(config, checkpoint, device=device)

color_mask = get_palette('concrete_damage_as_cityscapes')[1:]

det_result_dict = {}
num_classes = 4 # without background

for img_path in img_path_list:

    img_basename = osp.basename(img_path)

    det_result_dict[img_basename] = {}

    _, mask_output_high_res = inference_segmentor_sliding_window(model, img_path, color_mask, num_classes,
                                                                 window_size = 1024)
    _, mask_output_low_res = inference_segmentor_sliding_window(model, img_path, color_mask, num_classes,
                                                                 window_size=1024*4)

    mask_output = np.zeros_like(mask_output_high_res, dtype = np.bool)

    mask_output[:, :, 0] = connect_cracks(mask_output_high_res[:, :, 0])
    mask_output[:, :, 0] = connect_cracks(mask_output[:, :, 0])
    mask_output[:,:,0] = remove_small_obj(mask_output[:, :, 0])

    for num in range(1, num_classes):
        mask_output[:,:,num] = remove_small_obj(mask_output_low_res[:, :, num])


    img_result = imread(img_path)
    for num in range(num_classes):
        img_result[mask_output[:, :, num], :] = img_result[mask_output[:, :, num], :] * 0.3 + np.asarray(
            color_mask[num], dtype=np.float) * 0.6

    det_result_dict[img_basename]["anly_output"] = []

    for class_num in range(mask_output.shape[2]):

        dmg_result = mask_output[:, :, class_num]

        if class_num == 0:
            damage_type = 'crack'
        elif class_num == 1:
            damage_type = 'efflorescence'
        elif class_num == 2:
            damage_type = 'rebar'
        else:
            damage_type = 'spalling'

        if np.sum(dmg_result) > 0:

            labels = label(dmg_result)

            damage_region_prop = regionprops_table(labels, properties=('label', "bbox", 'area'))

            for label_num in tqdm(range(np.max(labels)), ascii=True, desc='Post processing for detected damages:'):
                if damage_region_prop['bbox-0'][label_num] > 50:

                    a_label = labels == label_num + 1

                    contours, _ = cv2.findContours(a_label.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    poly_step = 50  # if damage_type == 'crack' else 100

                    coords = []
                    for countour_num in range(0, len(contours[0]), poly_step):
                        coords.append([str(contours[0][countour_num][0][1]), str(contours[0][countour_num][0][0])])

                    dmg_info = {}
                    if class_num == 0:

                        min_row = int(damage_region_prop['bbox-0'][label_num])
                        max_row = int(damage_region_prop['bbox-2'][label_num])
                        min_col = int(damage_region_prop['bbox-1'][label_num])
                        max_col = int(damage_region_prop['bbox-3'][label_num])

                        a_label_skel = a_label[min_row: max_row, min_col: max_col].copy()

                        skel, distance = medial_axis(a_label_skel, return_distance=True)
                        dist_label = distance * skel

                        width = str(dist_label[np.nonzero(dist_label)].mean() * lenPerPixel)

                        dmg_info['damage_type'] = damage_type
                        dmg_info['id'] = str(label_num)
                        dmg_info['length'] = str(np.sum(skel) * lenPerPixel)
                        dmg_info['width'] = width
                        dmg_info['height'] = ""
                        dmg_info['area'] = ""
                        dmg_info['coords'] = coords

                    else:

                        min_row = int(damage_region_prop['bbox-0'][label_num])
                        max_row = int(damage_region_prop['bbox-2'][label_num])
                        min_col = int(damage_region_prop['bbox-1'][label_num])
                        max_col = int(damage_region_prop['bbox-3'][label_num])

                        dmg_info['damage_type'] = damage_type
                        dmg_info['id'] = str(label_num)
                        dmg_info['length'] = ""
                        dmg_info['width'] = str((max_col - min_col) * lenPerPixel)
                        dmg_info['height'] = str((max_row - min_row) * lenPerPixel)
                        dmg_info['area'] = str((max_col - min_col) * (max_row - min_row) * lenPerPixel * lenPerPixel)
                        dmg_info['coords'] = coords

                    det_result_dict[img_basename]["anly_output"].append(dmg_info)

    if (save_result == 'y') or (save_result == 'Y'):
        img_result_folder = osp.join(img_folder, 'result_img')
        Path(img_result_folder).mkdir(parents=True, exist_ok=True)
        img_result_filename = osp.join(img_result_folder, img_basename)
        imwrite(img_result_filename, img_result)

    json_folder = osp.join(img_folder,  'json')
    Path(json_folder).mkdir(parents=True, exist_ok=True)
    json_save_path = osp.join(json_folder, osp.splitext(img_basename)[0] + '.json')
    with open(json_save_path, 'w', encoding='utf8') as f:
        json.dump(det_result_dict, f, ensure_ascii=False)
