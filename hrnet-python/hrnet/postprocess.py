"""postprocess"""
import numpy as np
import math
import cv2
from .preprocess import get_affine_transform


def get_color(idx):
    """
    Get a color based on the given index.

    Args:
        idx (int): The index to calculate the color.

    Returns:
        tuple: A tuple representing the RGB values of the color.

    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class HRNetPostProcess(object):
    """
    HRNetPostProcess
    """

    def __init__(self, use_dark=True):
        """init"""
        self.use_dark = use_dark

    def flip_back(self, output_flipped, matched_parts):
        """
        flip_back
        """
        assert (
            output_flipped.ndim == 4
        ), "output_flipped should be [batch_size, num_joints, height, width]"

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped

    def get_max_preds(self, heatmaps):
        """get predictions from score maps

        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        """
        assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
        assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap, kernel):
        """gaussian_blur"""
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        """dark_parse"""
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (
                hm[py + 1][px + 1]
                - hm[py - 1][px + 1]
                - hm[py + 1][px - 1]
                + hm[py - 1][px - 1]
            )
            dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy ** 2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        """
        refer to https://github.com/ilovepose/DarkPose/lib/core/inference.py

        """
        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """

        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px],
                            ]
                        )
                        coords[n][p] += np.sign(diff) * 0.25
        preds = coords.copy()
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

        return preds, maxvals

    def __call__(self, output, center, scale):
        """entry"""
        preds, maxvals = self.get_final_preds(output, center, scale)
        return np.concatenate((preds, maxvals), axis=-1), np.mean(maxvals, axis=1)


def transform_preds(coords, center, scale, output_size):
    """transform_preds"""
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale * 200, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    """affine_transform"""
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def translate_to_ori_images(keypoint_result, batch_records):
    """translate_to_ori_images"""
    kpts = keypoint_result["keypoint"]
    scores = keypoint_result["score"]
    kpts[..., 0] += batch_records[:, 0:1]
    kpts[..., 1] += batch_records[:, 1:2]
    return kpts, scores


def filter_box(result, threshold):
    """filter_box"""
    np_boxes_num = result["boxes_num"]
    boxes = result["boxes"]
    start_idx = 0
    filter_boxes = []
    filter_num = []
    for i in range(len(np_boxes_num)):
        boxes_num = np_boxes_num[i]
        boxes_i = boxes[start_idx:start_idx + boxes_num, :]
        idx = boxes_i[:, 1] > threshold
        filter_boxes_i = boxes_i[idx, :]
        filter_boxes.append(filter_boxes_i)
        filter_num.append(filter_boxes_i.shape[0])
        start_idx += boxes_num
    boxes = np.concatenate(filter_boxes)
    filter_num = np.array(filter_num)
    filter_res = {"boxes": boxes, "boxes_num": filter_num}
    return filter_res


class KptPostProcess(object):
    """KptPostProcess"""

    def __call__(self, inputs, result):
        """entry"""
        output = {}
        imshape = np.array(inputs.shape[0:2][::-1])
        imshape = np.expand_dims(imshape, axis=0)
        center = np.round(imshape / 2.0)
        scale = imshape / 200.0
        keypoint_postprocess = HRNetPostProcess()
        result = np.array(result).reshape((1, 17, 64, 48))
        kpts, scores = keypoint_postprocess(result, center, scale)
        output["keypoint"] = kpts
        output["score"] = scores
        return output


