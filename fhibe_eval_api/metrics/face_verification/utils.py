# SPDX-License-Identifier: Apache-2.0
"""Utilities for the face verification task."""

import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm

from fhibe_eval_api.common.utils import create_folders
from fhibe_eval_api.metrics.face_verification.matlab_cp2tform import (
    get_similarity_transform_for_cv2,
)
from fhibe_eval_api.metrics.face_verification.mtcnn.mtcnn import mtcnn_model

# reference facial points, a list of coordinates (x,y)
# default reference facial points for crop_size = (112, 112); should adjust
# REFERENCE_FACIAL_POINTS accordingly for other crop_size
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156],
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):  # noqa: N818
    def __str__(self):  # noqa: D105
        return "In File {}:{}".format(__file__, super.__str__(self))


def get_reference_facial_points(
    output_size=None,
    inner_padding_factor=0.0,
    outer_padding=(0, 0),
    default_square=False,
):
    """Get reference facial points.

    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square:
                crop_size = (112, 112)
            else:
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding)
                = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    # print('\n===> get_reference_facial_points():')

    # print('---> Params:')
    # print('            output_size: ', output_size)
    # print('            inner_padding_factor: ', inner_padding_factor)
    # print('            outer_padding:', outer_padding)
    # print('            default_square: ', default_square)

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # print('---> default:')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    if (
        output_size
        and output_size[0] == tmp_crop_size[0]
        and output_size[1] == tmp_crop_size[1]
    ):
        # print('output_size == DEFAULT_CROP_SIZE {}: return default reference
        # points'.format(tmp_crop_size))
        return tmp_5pts

    if inner_padding_factor == 0 and outer_padding == (0, 0):
        if output_size is None:
            # print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise FaceWarpException(
                "No paddings to do, output_size must be None or {}".format(
                    tmp_crop_size
                )
            )

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException("Not (0 <= inner_padding_factor <= 1.0)")

    if (
        inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0
    ) and output_size is None:
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        # print('              deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException(
            "Not (outer_padding[0] < output_size[0]"
            "and outer_padding[1] < output_size[1])"
        )

    # 1) pad the inner region according inner_padding_factor
    # print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    # print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    # print('              crop_size = ', tmp_crop_size)
    # print('              size_bf_outer_pad = ', size_bf_outer_pad)

    if (
        size_bf_outer_pad[0] * tmp_crop_size[1]
        != size_bf_outer_pad[1] * tmp_crop_size[0]
    ):
        raise FaceWarpException(
            "Must have (output_size - outer_padding)"
            "= some_scale * (crop_size * (1.0 + "
            "inner_padding_factor)"
        )

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    # print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
    #    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
    #    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    # print('---> STEP3: add outer_padding to make output_size')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # print('===> end get_reference_facial_points\n')

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
    """

    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    #    #print(('src_pts_:\n' + str(src_pts_))
    #    #print(('dst_pts_:\n' + str(dst_pts_))

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    #    #print(('np.linalg.lstsq return A: \n' + str(A))
    #    #print(('np.linalg.lstsq return res: \n' + str(res))
    #    #print(('np.linalg.lstsq return rank: \n' + str(rank))
    #    #print(('np.linalg.lstsq return s: \n' + str(s))

    if rank == 3:
        tfm = np.float32([[A[0, 0], A[1, 0], A[2, 0]], [A[0, 1], A[1, 1], A[2, 1]]])
    elif rank == 2:
        tfm = np.float32([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]])

    return tfm


def warp_and_crop_face(
    src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type="smilarity"
):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding, default_square
            )

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException("reference_pts.shape must be (K,2) or (2,K) and K>2")

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException("facial_pts.shape must be (K,2) or (2,K) and K>2")

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    #    #print('--->src_pts:\n', src_pts
    #    #print('--->ref_pts\n', ref_pts

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException("facial_pts and reference_pts must have the same shape")

    if align_type == "cv2_affine":
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    #        #print(('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type == "affine":
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    #        #print(('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
    #        #print(('get_similarity_transform_for_cv2() returns tfm=\n' + str(tfm))

    #    #print('--->Transform matrix: '
    #    #print(('type(tfm):' + str(type(tfm)))
    #    #print(('tfm.dtype:' + str(tfm.dtype))
    #    #print( tfm

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img


def align_faces(
    img_filepaths: List[str],
    aligned_img_filepaths: List[str],
    crop_size: int,
    cuda: bool,
) -> List[bool]:
    if len(img_filepaths) != len(aligned_img_filepaths):
        raise ValueError("len(img_filepaths) != len(aligned_img_filepaths)")

    scale = crop_size / 112.0
    reference = get_reference_facial_points(default_square=True) * scale

    detector = mtcnn_model(
        implementation="face_evolve",
        min_face_size=20,
        thresholds=(0.6, 0.7, 0.8),
        nms_thresholds=(0.7, 0.7, 0.7),
        cuda=False,
    )

    success = []
    with tqdm(total=len(img_filepaths), desc="Align faces in images") as pbar:
        for img_filepath, aligned_img_filepath in zip(
            img_filepaths, aligned_img_filepaths
        ):
            if os.path.isfile(aligned_img_filepath):
                success.append(True)
                continue

            try:
                img = Image.open(img_filepath).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError):
                success.append(False)
                continue

            _, landmarks = detector.detect_faces(img)  # type: ignore

            if landmarks is None:
                success.append(False)
                continue
            elif len(landmarks) == 0:
                success.append(False)
                continue

            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(
                np.array(img),
                facial5points,
                reference,
                crop_size=(crop_size, crop_size),
            )

            img_warped = Image.fromarray(warped_face)

            create_folders(filepath=aligned_img_filepath)
            img_warped.save(aligned_img_filepath)
            success.append(True)
            pbar.update(1)
    return success


def calculate_roc(
    thresholds, embeddings1, embeddings2, dist, actual_issame, nrof_folds=10
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _, _ = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            (
                tprs[fold_idx, threshold_idx],
                fprs[fold_idx, threshold_idx],
                _,
                _,
                _,
            ) = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc, is_fp, is_fn


def calculate_val(
    thresholds, embeddings1, embeddings2, dist, actual_issame, far_target, nrof_folds=10
):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set]
            )
        # if np.max(far_train) >= far_target:
        if np.max(far_train) >= far_target and np.min(far_train) <= far_target:
            # remove duplicates
            # print("far_train", far_train)
            far_train, unique_idxs = np.unique(far_train, return_index=True)
            # print("unique far_train", far_train)
            thresholds = np.array([thresholds[x] for x in unique_idxs])
            f = interpolate.interp1d(far_train, thresholds, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set]
        )

    val_mean = np.nanmean(val)
    far_mean = np.nanmean(far)
    val_std = np.nanstd(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(true_accept) / float(n_same) if n_same != 0 else np.nan
    far = float(false_accept) / float(n_diff) if n_diff != 0 else np.nan

    return val, far


def evaluate(embeddings1, embeddings2, dist, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(-1, 1, 0.01)
    tpr, fpr, accuracy, fp, fn = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        dist,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
    )
    thresholds = np.arange(-1, 1, 0.001)
    val, val_std, far = calculate_val(
        thresholds,
        embeddings1,
        embeddings2,
        dist,
        np.asarray(actual_issame),
        1e-3,
        nrof_folds=nrof_folds,
    )
    return tpr, fpr, accuracy, val, val_std, far, fp, fn


def _generate_positive_pairs(filepaths):
    local_pairs = set()
    for file1, file2 in itertools.combinations(filepaths, 2):
        local_pairs.add(tuple(sorted([file1, file2])))
    return local_pairs


def get_positive_pairs(subset_df, filepath_col, person_col, n_pairs):
    df = subset_df[[filepath_col, person_col]]

    same_pairs = set()
    with ProcessPoolExecutor() as executor:
        futures = []
        for _, group in df.groupby(person_col):
            filepaths = group[filepath_col].tolist()
            futures.append(executor.submit(_generate_positive_pairs, filepaths))

        for future in as_completed(futures):
            same_pairs.update(future.result())

    same_pairs = list(same_pairs)
    same_pairs = sorted(same_pairs)

    if n_pairs is not None:
        same_pairs = random.sample(same_pairs, min(n_pairs, len(same_pairs)))

    return same_pairs


def _generate_negative_pairs(start, end, persons, filepaths):
    local_pairs = set()
    for i in range(start, end):
        for j in range(i + 1, len(filepaths)):
            if persons[i] != persons[j]:
                pair = tuple(sorted((filepaths[i], filepaths[j])))
                local_pairs.add(pair)
    return local_pairs


def get_negative_pairs(subset_df, filepath_col, person_col, n_pairs):

    filepaths = subset_df[filepath_col].values
    persons = subset_df[person_col].values

    num_chunks = os.cpu_count()
    chunk_size = len(filepaths) // min(len(filepaths), num_chunks)
    chunks = [
        (i * chunk_size, (i + 1) * chunk_size if i < num_chunks - 1 else len(filepaths))
        for i in range(num_chunks)
    ]

    unique_pairs = set()
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_generate_negative_pairs, start, end, persons, filepaths)
            for start, end in chunks
        ]

        for future in as_completed(futures):
            unique_pairs.update(future.result())

    diff_pairs = list(unique_pairs)
    diff_pairs = sorted(diff_pairs)

    diff_pairs = random.sample(diff_pairs, min(n_pairs, len(diff_pairs)))
    return diff_pairs
