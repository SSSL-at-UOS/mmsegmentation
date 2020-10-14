
import mmcv
import cv2
import os
import glob
import numpy as np
import slidingwindow as sw
import math

from mmseg.apis import inference_segmentor
from tqdm import tqdm
from slidingwindow import SlidingWindow
# from mmdet.apis import init_detector, inference_detector
from skimage.measure import label, regionprops_table


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        imageBGR = cv2.imdecode(n, flags)
        return cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(e)
        return None


def imwrite(filename, imageRGB, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        imageBGR = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        result, n = cv2.imencode(ext, imageBGR, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
                return False

    except Exception as e:
        print(e)
        return False

def inference_segmentor_sliding_window(model, input_img, color_mask, num_classes,
                                       window_size = 1024, overlap_ratio = 0.1, area_thr = 100):


    '''
    :param model: is a mmdetection model object
    :param input_img : str or numpy array
                    if str, run imread from input_img
    :param score_thr: is float number between 0 and 1.
                   Bounding boxes with a confidence higher than score_thr will be displayed,
                   in 'img_result' and 'mask_output'.
    :param window_size: is a subset size to be detected at a time.
                        default = 1024, integer number
    :param overlap_ratio: is a overlap size.
                        If you overlap sliding windows by 50%, overlap_ratio is 0.5.

    :return: img_result
    :return: mask_output

    '''

    # color mask has to be updated for multiple-class object detection
    if isinstance(input_img, str) :
        img = imread(input_img)
    else :
        img = input_img

    # Generate the set of windows, with a 256-pixel max window size and 50% overlap
    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, window_size, overlap_ratio)
    mask_output = np.zeros((img.shape[0], img.shape[1], num_classes), dtype=np.uint8)

    if isinstance(input_img, str) :
        tqdm_window = tqdm(windows, ascii=True, desc='inference by sliding window on ' + os.path.basename(input_img))
    else :
        tqdm_window = tqdm(windows, ascii=True, desc='inference by sliding window ')

    for window in tqdm_window :
        # Add print option for sliding window detection

        img_subset = img[window.indices()]
        subset_size = (img_subset.shape[1], img_subset.shape[0])
        img_subset_resize = cv2.resize(img_subset, (1024, 1024))
        results = inference_segmentor(model, img_subset_resize)[0]
        results = cv2.resize(np.asarray(results, dtype=np.uint8), subset_size, interpolation=cv2.INTER_NEAREST )
        results_onehot = (np.arange(num_classes) == results[...,None]-1).astype(int)

        mask_output[window.indices()] = mask_output[window.indices()] + results_onehot

    mask_output[mask_output > 1] = 1

    mask_output_bool = mask_output.astype(np.bool)

        # Add colors to detection result on img
    img_result = img
    for num in range(num_classes) :
        img_result[mask_output_bool[:,:,num], :] = img_result[mask_output_bool[:,:,num],:] * 0.3 + np.asarray(color_mask[num], dtype = np.float) * 0.6

    return img_result, mask_output


# def inference_detector_sliding_window(model, input_img, color_mask,
#                                       score_thr = 0.1, window_size = 1024, overlap_ratio = 0.5,):
#
#
#     '''
#     :param model: is a mmdetection model object
#     :param input_img : str or numpy array
#                     if str, run imread from input_img
#     :param score_thr: is float number between 0 and 1.
#                    Bounding boxes with a confidence higher than score_thr will be displayed,
#                    in 'img_result' and 'mask_output'.
#     :param window_size: is a subset size to be detected at a time.
#                         default = 1024, integer number
#     :param overlap_ratio: is a overlap size.
#                         If you overlap sliding windows by 50%, overlap_ratio is 0.5.
#
#     :return: img_result
#     :return: mask_output
#
#     '''
#
#     # color mask has to be updated for multiple-class object detection
#     if isinstance(input_img, str) :
#         img = imread(input_img)
#     else :
#         img = input_img
#
#     # Generate the set of windows, with a 256-pixel max window size and 50% overlap
#     windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, window_size, overlap_ratio)
#     mask_output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#
#     if isinstance(input_img, str) :
#         tqdm_window = tqdm(windows, ascii=True, desc='inference by sliding window on ' + os.path.basename(input_img))
#     else :
#         tqdm_window = tqdm(windows, ascii=True, desc='inference by sliding window ')
#
#
#     for window in tqdm_window :
#         # Add print option for sliding window detection
#         img_subset = img[window.indices()]
#         img_shorter_axis_length = np.min((img_subset.shape[0], img_subset.shape[1]))
#
#         scale_percent = 1024 / img_shorter_axis_length  # percent of original size
#         width = int(img_subset.shape[1] * scale_percent)
#         height = int(img_subset.shape[0] * scale_percent)
#         dim = (width, height)
#         img_subset_resize = cv2.resize(img_subset, dim)
#
#         results = inference_detector(model, img_subset_resize)
#         bbox_result, segm_result = results
#         mask_sum = np.zeros((img_subset_resize.shape[0], img_subset_resize.shape[1]), dtype=np.uint8)
#         bboxes = np.vstack(bbox_result) # bboxes
#
#         # draw segmentation masks
#         if segm_result is not None:
#             segms = mmcv.concat_list(segm_result)
#             inds = np.where(bboxes[:, -1] > score_thr)[0]
#
#             for i in inds:
#                 mask = segms[i].astype(np.uint8)
#                 mask_sum = mask_sum + mask
#
#
#         dim = (img_subset.shape[1], img_subset.shape[0])
#         mask_sum = mask_sum.astype(np.uint8)
#         mask_sum = cv2.resize(mask_sum, dim)
#
#         mask_output[window.indices()] = mask_output[window.indices()] + mask_sum
#
#     mask_output[mask_output > 1] = 1
#
#     mask_output_bool = mask_output.astype(np.bool)
#
#     # Add colors to detection result on img
#     img_result = img
#     img_result[mask_output_bool, :] = img_result[mask_output_bool,:] * 0.3 + color_mask * 0.6
#
#     return img_result, mask_output
#

def connect_cracks(mask_output, epsilon = 200):
    '''
    :param mask_output: a numpy uint8 variable
    :param epsilon: distance between cracks to be connected
    :return: connect_mask_output : crack-connection result
    '''

    '''
    To-dos : 
    1 . Add iteration optionF
    2 . Add connection option considering a direction of a crack
    with the direction of ellipse of each crack 
    '''

    # label each crack
    labels, num = label(mask_output, connectivity=2, return_num=True)
    # get information of each crack area
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords', 'orientation'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']

    crack_region_table['is_horizontal'] = width > height

    connecting_directions = ['x_axis', 'y_axis']
    connect_line_img = np.zeros_like(mask_output, dtype=np.uint8)

    for connecting_direction in connecting_directions:

        e2_list = []
        e1_list = []

        for crack_num, crack_region in enumerate(crack_region_table['label']):

            min_row = crack_region_table['bbox-0'][crack_num]
            min_col = crack_region_table['bbox-1'][crack_num]
            max_row = crack_region_table['bbox-2'][crack_num] - 1
            max_col = crack_region_table['bbox-3'][crack_num] - 1

            if crack_region_table['is_horizontal'][crack_num]:
                # max col / min col
                col = crack_region_table['coords'][crack_num][:, 1]

                e2 = crack_region_table['coords'][crack_num][np.argwhere(col == max_col), :][-1][0]
                e1 = crack_region_table['coords'][crack_num][np.argwhere(col == min_col), :][0][0]

                if connecting_direction == 'y_axis' and e2[0] < e1[0]:
                    e2, e1 = e1, e2

                e2_list.append(e2)
                e1_list.append(e1)

            else:
                # max row / min row
                row = crack_region_table['coords'][crack_num][:, 0]

                e2 = crack_region_table['coords'][crack_num][np.argwhere(row == max_row), :][-1][0]
                e1 = crack_region_table['coords'][crack_num][np.argwhere(row == min_row), :][0][0]

                if connecting_direction == 'x_axis' and e2[1] < e1[1]:
                    e2, e1 = e1, e2

                e2_list.append(e2)
                e1_list.append(e1)

        crack_region_table['e2'] = e2_list
        crack_region_table['e1'] = e1_list


        n = len(crack_region_table['label'])
        color = (1)  # binary image


        for num_e2, e2 in enumerate(crack_region_table['e2']):

            connect_candidates_e2 = []
            connect_candidates_e1 = []
            distance_list = []

            for num_e1, e1 in enumerate(crack_region_table['e1']):

                if num_e2 != num_e1:
                    d = np.subtract(e1, e2)
                    distance = np.sqrt(d[0] ** 2 + d[1] ** 2)


                    vector_1 = np.asarray(crack_region_table['e1'][num_e2] - e2, dtype=np.float64)
                    vector_2 = np.asarray(e2 - e1, dtype=np.float64)

                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle_1 = np.arccos(dot_product)


                    vector_1 = np.asarray(crack_region_table['e1'][num_e2] - e2, dtype=np.float64)
                    vector_2 = np.asarray(e1 - crack_region_table['e2'][num_e1], dtype=np.float64)

                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle_2 = np.arccos(dot_product)


                    if (distance < epsilon) and (angle_1 < 1.5 / 2) and (angle_2 < 1.5 / 2):
                        distance_list.append(distance)
                        connect_candidates_e2.append(tuple(e2[::-1]))
                        connect_candidates_e1.append(tuple(e1[::-1]))

                    if (distance < epsilon/3):
                        distance_list.append(distance)
                        connect_candidates_e2.append(tuple(e2[::-1]))
                        connect_candidates_e1.append(tuple(e1[::-1]))

            if distance_list :
                connect_idx = np.argmin(distance_list)
                connect_e2 = connect_candidates_e2[connect_idx]
                connect_e1 = connect_candidates_e1[connect_idx]
                connect_line_img = cv2.line(connect_line_img, connect_e2, connect_e1, color, 2)

    mask_output = mask_output + connect_line_img
    mask_output[mask_output > 1] = 1

    return mask_output

def remove_small_obj(mask_output, threshold = 300):
    '''
    :param mask_output: a numpy uint8 variable
    :param threshold: cracks of which length is under thershold will be removed.
    :return: mask_output : crack mask after thresholding
    '''

    labels, num = label(mask_output, connectivity=2, return_num=True)
    crack_region_table = regionprops_table(labels, properties=('label', 'bbox', 'coords'))

    width = crack_region_table['bbox-3'] - crack_region_table['bbox-1']
    height = crack_region_table['bbox-2'] - crack_region_table['bbox-0']
    crack_region_table['diagonal_length'] = np.sqrt(height**2 + width**2)

    for crack_num in range(len(crack_region_table['label'])):
        if crack_region_table['diagonal_length'][crack_num] < threshold :
            for c in crack_region_table['coords'][crack_num]:
                mask_output[c[0], c[1]] = 0

    return mask_output


class AverageMeter(object):
    # This function is imported from https://github.com/hszhao/semseg/blob/master/util/util.py
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def comparison_operator(img, thr, ind='>'):
    import operator
    if ind == '==':
        return operator.eq(img, thr)
    elif ind == '<':
        return operator.lt(img, thr)
    elif ind == '>':
        return operator.gt(img, thr)
    elif ind == '!=':
        return operator.ne(img, thr)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # This function is imported from https://github.com/hszhao/semseg/blob/master/util/util.py
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def cal_acc(data_list, pred_folder, classes, names):
    # This function is imported from https://github.com/hszhao/semseg/blob/master/util/util.py
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info(
            'Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name + '.png',
                                                                        accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))


def sw_generate_specific_width_height(data, width_height, dimOrder, maxWindowSize, overlapPercent, transforms=[]):
    """
    Generates a set of sliding windows for the specified dataset.
    """

    # Determine the dimensions of the input data
    width, height = width_height
    lastX, lastY = data.shape[:-1]

    # Generate the windows
    return generateForSize_specific_width_height(width, height, lastX, lastY, dimOrder, maxWindowSize, overlapPercent,
                                                 transforms)


def generateForSize_specific_width_height(width, height, lastX, lastY, dimOrder, maxWindowSize, overlapPercent,
                                          transforms=[]):
    """
    Generates a set of sliding windows for a dataset with the specified dimensions and order.
    """


    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    windowSizeX = min(maxWindowSize, width)
    windowSizeY = min(maxWindowSize, height)

    # Compute the window overlap and step size
    windowOverlapX = int(math.floor(windowSizeX * overlapPercent))
    windowOverlapY = int(math.floor(windowSizeY * overlapPercent))
    stepSizeX = windowSizeX - windowOverlapX
    stepSizeY = windowSizeY - windowOverlapY

    # Determine how many windows we will need in order to cover the input data
    #     lastX = width - windowSizeX
    #     lastY = height - windowSizeY
    xOffsets = list(range(0, lastX + 1, stepSizeX))
    yOffsets = list(range(0, lastY + 1, stepSizeY))

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    if len(xOffsets) == 0 or xOffsets[-1] != lastX:
        xOffsets.append(lastX)
    if len(yOffsets) == 0 or yOffsets[-1] != lastY:
        yOffsets.append(lastY)

    # Generate the list of windows
    windows = []
    for xOffset in xOffsets:
        for yOffset in yOffsets:
            for transform in [None] + transforms:
                windows.append(SlidingWindow(
                    x=xOffset,
                    y=yOffset,
                    w=windowSizeX,
                    h=windowSizeY,
                    dimOrder=dimOrder,
                    transform=transform
                ))

    return windows
