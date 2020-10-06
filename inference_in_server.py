# import required libraries
import numpy as np
import mmcv
import cv2
import os
import glob
import sys
import datetime
import argparse
import json

import slidingwindow as sw

from pathlib import Path
from collections import OrderedDict
from multiprocessing import Pool

from mmdet.apis import init_detector, inference_detector
from skimage.measure import label, regionprops_table, find_contours
from skimage.morphology import medial_axis, skeletonize

sys.path.append('..')
from shm_tools.shm_utils import imread, imwrite, inference_detector_sliding_window, connect_cracks, remove_cracks
from shm_tools.SlidingWindow import generateForNumberOfWindows

def sort_dict(dict, key_order):
    dict = OrderedDict(dict)

    for k in key_order:
        dict.move_to_end(k)

    return dict


def find_alligator_crack(mask_output, struct_type, lenPerPixel):
    if struct_type == 'TN':
        window_size = 1024
        windows = sw.generate(mask_output, sw.DimOrder.HeightWidthChannel, window_size, overlapPercent=0)
        # call sw by window size
    elif struct_type == 'BR':
        window_size = 1024
        windows = sw.generate(mask_output, sw.DimOrder.HeightWidthChannel, window_size, overlapPercent=0)

    elif struct_type == 'BP':
        windowCount = (4, 2)
        windows = generateForNumberOfWindows(mask_output, sw.DimOrder.HeightWidthChannel, windowCount, overlapPercent=0,
                                             transforms=[])

    img_result_alg_crack = np.zeros_like(mask_output)

    for num, window in enumerate(windows):
        img_subset = mask_output[window.indices()]
        labels, _ = label(img_subset, connectivity=2, return_num=True)

        if np.max(labels) > 2:

            crack_skel = skeletonize(img_subset)
            crack_length = np.sum(crack_skel) * lenPerPixel
            crack_area = crack_length * 250 / 100
            window_area = (img_subset.shape[0] * lenPerPixel) * (img_subset.shape[1] * lenPerPixel) / 100
            crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords', 'orientation'))

            if (np.std(crack_region_table['orientation']) > 20 * 0.0174533) and (crack_area > 0.6 * window_area):
                img_result_alg_crack[window.indices()] = 1

    return img_result_alg_crack


def write_cordList(imprXcord,
                   imprYcord,
                   imprCnterCord,
                   imprTypeCd,
                   cordTypeCd="2",
                   imprWdth="",
                   imprLnth="",
                   imprBrdthVal="",
                   imprQntt="", ):
    cordList = {}

    cordList["cordTypeCd"] = cordTypeCd
    cordList["imprXcord"] = imprXcord
    cordList["imprYcord"] = imprYcord
    cordList["imprCnterCord"] = imprCnterCord

    if imprTypeCd == "crack":
        cordList["imprTypeCd"] = "01"
    elif imprTypeCd == "ali_crack":
        cordList["imprTypeCd"] = "02"
    elif imprTypeCd == "delimination":
        cordList["imprTypeCd"] = "03"
    elif imprTypeCd == "spll":
        cordList["imprTypeCd"] = "04"
    elif imprTypeCd == "effl":
        cordList["imprTypeCd"] = "05"

    cordList["imprWdth"] = imprWdth
    cordList["imprLnth"] = imprLnth
    cordList["imprBrdthVal"] = imprBrdthVal
    cordList["imprQntt"] = imprQntt

    return cordList

# read inference configuration json file
parser = argparse.ArgumentParser()
parser.add_argument("inference_config")
args = parser.parse_args()


with open(args.inference_config) as f:
    inference_config_from_ui = json.load(f)

with open('shm_tools/inference_config_in_module.json') as f:
    inference_config_in_module = json.load(f)

# set img path list
img_folder =inference_config_from_ui['anlyTargetPath']
img_path_list = glob.glob(os.path.join(img_folder, '*.jpg')) + glob.glob(os.path.join(img_folder, '*.JPG'))
img_path_list = sorted(img_path_list)

# set result path
result_save_folder = inference_config_from_ui["anlyResultPath"]

damage_detection_output = {}
damage_detection_output["ptanFcltsCd"] = inference_config_from_ui["ptanFcltsCd"]
damage_detection_output["anlyDataId"] = inference_config_from_ui["anlyDataId" ]
damage_detection_output["pctrList"] = []

# loop through imgs in the list
# detection and post processing for crack

struct_type = inference_config_from_ui["ptanFcltsCd"]
lenPerPixel = np.float64(inference_config_from_ui['pctrPtgrDstne'])

# Set color mask
color_mask = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 255, 255],
                       [255, 0, 255],
                       ], dtype=np.uint8)

pctrList_key_order = ['anlyPctrId', 'anlyPctrNm', 'dfctCnt', 'strtDttm', 'endDttm', "cordList"]
result_key_order = ['ptanFcltsCd', 'anlyDataId', 'rsltCd', 'rsltCtnt', 'pctrList']




config = glob.glob(os.path.join('shm_work_dirs', struct_type, damage_type, '*.py'))[0]
checkpoint = glob.glob(os.path.join('shm_work_dirs', struct_type, damage_type, '*.pth'))[0]

# load the model on GPU
device = 'cuda:0'
model = init_detector(config, checkpoint, device=device)

# inference for crack
for num, img_path in enumerate(img_path_list):

    # if dmg_num == 0 :
    pctrList = {}
    pctrList["strtDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pctrList["anlyPctrId"] = str(num + 1)
    pctrList["anlyPctrNm"] = os.path.basename(img_path)
    damage_type == 'crack'

    # if dmg_num == 0 :
    #     window_size = 1024 +256
    #     color_mask_input = color_mask[0]
    #     overlap_ratio = 0.1
    #     score_thr = 0.2
    #
    # elif damage_type == 'spll':
    #     window_size = 1024 * 6
    #     color_mask_input = color_mask[3]
    #     overlap_ratio = 0.5
    #     score_thr = 0.5
    #
    # elif damage_type == 'effl':
    #     window_size = ( 1024 + 256 ) * 2
    #     color_mask_input = color_mask[1]
    #     overlap_ratio = 0.5
    #     score_thr = 0.9

    _, mask_output = inference_detector_sliding_window(model,
                                                       img_path,
                                                       color_mask_input,
                                                       score_thr=0.2,
                                                       window_size=window_size,
                                                       overlap_ratio=overlap_ratio)

    org_img = imread(img_path)

    if np.sum(mask_output) == 0:
        print(img_path + ' has no detection result')
        if dmg_num == 0 :
            # imwrite(os.path.join(result_save_folder, os.path.basename(img_path)), org_img)
            pctrList['cordList'] = []
            damage_detection_output["pctrList"].append(pctrList)

    elif np.sum(mask_output) > 0:

        mask_output = connect_cracks(mask_output)
        mask_output = connect_cracks(mask_output)
        mask_output = remove_cracks(mask_output, threshold=500)
        mask_output[alg_crack == 1] = 0

        labels = label(mask_output)

        pctrList['cordList'] = []

        if np.sum(labels) > 0 :

            damage_region_prop = regionprops_table(labels, properties=('label', 'centroid', "bbox"))

            for label_num in range(np.max(labels)):

                a_label = labels == label_num + 1

                contours, _ = cv2.findContours(a_label.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                imprXcord = str()
                imprYcord = str()

                poly_step = 50 if damage_type == 'crack' else 80

                for countour_num in range(0, len(contours[0]), poly_step):
                    imprYcord = imprYcord + str(contours[0][countour_num][0][1]) + ','
                    imprXcord = imprXcord + str(contours[0][countour_num][0][0]) + ','

                imprYcord = imprYcord + str(contours[0][0][0][1])
                imprXcord = imprXcord + str(contours[0][0][0][0])

                if dmg_num == 0 :

                    imprCnterCord = str(damage_region_prop['centroid-1'][label_num]) + ',' + str(
                        damage_region_prop['centroid-0'][label_num])

                    min_row = int(damage_region_prop['bbox-0'][label_num])
                    max_row = int(damage_region_prop['bbox-2'][label_num])
                    min_col = int(damage_region_prop['bbox-1'][label_num])
                    max_col = int(damage_region_prop['bbox-3'][label_num])

                    a_label_skel = a_label[min_row: max_row, min_col: max_col].copy()

                    skel, distance = medial_axis(a_label_skel, return_distance=True)
                    dist_label = distance * skel

                    if struct_type == 'TN' :
                        hist_width = np.histogram(org_img[a_label == 1])[1][-1] - np.histogram(org_img[a_label == 1])[1][0]
                        if hist_width > 50 :
                            width = lenPerPixel
                        elif (hist_width > 30) and (hist_width < 50 ):
                            width = lenPerPixel*0.5
                        elif (hist_width > 10) and (hist_width < 30 ):
                            width = lenPerPixel*0.3
                        elif (hist_width < 10 ):
                            width = lenPerPixel*0.1

                    else :
                        width = str(dist_label[np.nonzero(dist_label)].mean() * lenPerPixel / 4)

                    cordList = write_cordList(imprXcord,
                                              imprYcord,
                                              imprCnterCord,
                                              imprTypeCd='crack',
                                              imprWdth=width,
                                              imprLnth=str(np.sum(skel) * lenPerPixel),
                                              imprQntt=str(np.sum(skel) * lenPerPixel)
                                              )

                else:

                    imprCnterCord = str(damage_region_prop['centroid-1'][label_num]) + ',' + str(damage_region_prop['centroid-0'][label_num])

                    min_row = int(damage_region_prop['bbox-0'][label_num])
                    max_row = int(damage_region_prop['bbox-2'][label_num])
                    min_col = int(damage_region_prop['bbox-1'][label_num])
                    max_col = int(damage_region_prop['bbox-3'][label_num])

                    imprQntt = str((max_col - min_col) * (max_row - min_row) * lenPerPixel * lenPerPixel)
                    imprLnth = str((max_col - min_col) * lenPerPixel)
                    imprBrdthVal =str((max_row - min_row) * lenPerPixel)

                    cordList = write_cordList(imprXcord,
                                              imprYcord,
                                              imprCnterCord,
                                              imprTypeCd=damage_type,
                                              imprBrdthVal=imprBrdthVal,
                                              imprLnth = imprLnth,
                                              imprQntt=imprQntt
                                              )

                pctrList['cordList'].append(cordList)


            if np.sum(alg_crack) > 0:

                labels = label(alg_crack)
                damage_region_prop = regionprops_table(labels, properties=('label', 'centroid', "bbox"))

                damage_type = "ali_crack"

                for label_num in range(np.max(labels)):

                    a_label = labels == label_num + 1

                    contours, _ = cv2.findContours(a_label.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    imprXcord = str()
                    imprYcord = str()

                    poly_step = 50 if damage_type == 'crack' else 80

                    for countour_num in range(0, len(contours[0]), poly_step):
                        imprYcord = imprYcord + str(contours[0][countour_num][0][1]) + ','
                        imprXcord = imprXcord + str(contours[0][countour_num][0][0]) + ','

                    imprYcord = imprYcord + str(contours[0][0][0][1])
                    imprXcord = imprXcord + str(contours[0][0][0][0])

                    imprCnterCord = str(damage_region_prop['centroid-1'][label_num]) + ',' + str(damage_region_prop['centroid-0'][label_num])

                    min_row = int(damage_region_prop['bbox-0'][label_num])
                    max_row = int(damage_region_prop['bbox-2'][label_num])
                    min_col = int(damage_region_prop['bbox-1'][label_num])
                    max_col = int(damage_region_prop['bbox-3'][label_num])

                    imprQntt = str((max_col - min_col) * (max_row - min_row) * lenPerPixel * lenPerPixel)
                    imprLnth = str((max_col - min_col) * lenPerPixel)
                    imprBrdthVal =str((max_row - min_row) * lenPerPixel)

                    cordList = write_cordList(imprXcord,
                                              imprYcord,
                                              imprCnterCord,
                                              imprTypeCd=damage_type,
                                              imprBrdthVal=imprBrdthVal,
                                              imprLnth = imprLnth,
                                              imprQntt=imprQntt
                                              )

                    pctrList['cordList'].append(cordList)

        if inference_config_from_ui["anlyPctrYn"] == 'y' or inference_config_from_ui["anlyPctrYn"] == 'Y':
            if dmg_num == 0:
                img_result = imread(img_path)
                Path(result_save_folder).mkdir(parents=True, exist_ok=True)

                mask_output_bool = mask_output.astype(np.bool)
                img_result[mask_output_bool, :] = img_result[mask_output_bool, :] * 0.3 + color_mask_input * 0.6

                alg_crack_bool =  alg_crack.astype(np.bool)
                img_result[alg_crack_bool, :] = img_result[alg_crack_bool, :] * 0.9 + color_mask_input * 0.1

                imwrite(os.path.join(result_save_folder, os.path.basename(img_path)), img_result)

            else:
                if os.path.isfile(os.path.join(result_save_folder, os.path.basename(img_path))) :
                    img_result = imread(os.path.join(result_save_folder, os.path.basename(img_path)))
                else :
                    img_result = imread(img_path)

                mask_output_bool = mask_output.astype(np.bool)
                img_result[mask_output_bool, :] = img_result[mask_output_bool, :] * 0.3 + color_mask_input * 0.6
                imwrite(os.path.join(result_save_folder, os.path.basename(img_path)), img_result)

        if dmg_num == 0 :
            pctrList["dfctCnt"] = str(len(pctrList['cordList']))
            pctrList["endDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            pctrList = sort_dict(pctrList, pctrList_key_order)


            damage_detection_output["pctrList"].append(pctrList)

        else:
            pctrList["dfctCnt"] = str(len(pctrList['cordList']))
            pctrList["endDttm"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            pctrList = sort_dict(pctrList, pctrList_key_order)
            damage_detection_output["pctrList"][num] = pctrList

damage_detection_output["rsltCd"] = "S"
damage_detection_output["rsltCtnt"] = ""

damage_detection_output = OrderedDict(damage_detection_output)

damage_detection_output = sort_dict(damage_detection_output, result_key_order)

save_path = os.path.join(inference_config_from_ui["anlyResultPath"], 'result.json')
Path(inference_config_from_ui["anlyResultPath"]).mkdir(parents=True, exist_ok=True)

with open(save_path, 'w', encoding='utf8') as f:
    json.dump(damage_detection_output, f, ensure_ascii=False)



