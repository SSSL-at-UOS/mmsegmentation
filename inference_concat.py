import json
import csv
import argparse
import cv2
import numpy as np
import os.path as osp


from shm_tools.shm_utils import imread, imwrite
from subprocess import call
from pathlib import Path

# read inference configuration json file
parser = argparse.ArgumentParser()
parser.add_argument("img_folder_name")
parser.add_argument("pixel_len")
parser.add_argument("save_result")
args = parser.parse_args()

img_folder = args.img_folder_name
lenPerPixel = np.float32(args.pixel_len)
save_result = args.save_result

call(['python', 'inference_patch.py', str(img_folder), str(lenPerPixel), str(save_result)])

header_info_path = osp.join(img_folder, 'header', 'info.txt')
header_img_path = osp.join(img_folder, 'header', 'img.jpg')
json_folder = osp.join(img_folder, 'json')
result_img_folder = osp.join(img_folder, 'result_img')
concat_folder = osp.join(img_folder, 'concat')

header_info = list(csv.reader(open(header_info_path, 'rt'), delimiter='\t'))
concat_json = {}
concat_json['anly_output'] = []
concat_width = int(header_info[-1][3]) + int(header_info[-1][1])
concat_height = int(header_info[-1][4]) + int(header_info[-1][2])
concat_img = np.zeros((concat_height, concat_width, 3), dtype=np.uint8)

for header_ in header_info[1:]:

    header = {}
    header['name'] = header_[0]
    header['width'] = int(header_[1])
    header['height'] = int(header_[2])
    header['start_x_point'] = int(header_[3])
    header['start_y_point'] = int(header_[4])

    json_filename = osp.join(json_folder, osp.splitext(header['name'])[0] + '.json')
    result_filename = osp.join(result_img_folder, header['name'])

    with open(json_filename) as f :
        json_info_ = json.load(f)

    json_info = json_info_[header['name']]

    for anly_info in json_info['anly_output']:
        concat_coords = []
        for coord_ in anly_info['coords'] :
            coord = np.asarray(coord_, dtype=np.int64)
            coord[0] += header['start_y_point']
            coord[1] += header['start_x_point']
            concat_coords.append(coord.tolist())
        anly_info['coords'] = concat_coords
        concat_json['anly_output'].append(anly_info)

    concat_img[header['start_y_point']:header['start_y_point']+header['height'],
    header['start_x_point']:header['start_x_point']+header['width'],:] = imread(result_filename)

Path(concat_folder).mkdir(parents=True, exist_ok=True)
header_img = imread(header_img_path)
concat_img_resize = cv2.resize(concat_img, (header_img.shape[:2][::-1]))
imwrite(osp.join(concat_folder, 'img.jpg'), concat_img_resize)
with open(osp.join(concat_folder, 'info.json'), 'w') as json_file :
    json.dump(concat_json, json_file)