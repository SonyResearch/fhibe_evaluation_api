# SPDX-License-Identifier: Apache-2.0
"""Module containing metric utils.

This module contains helper function for metrics.
"""

import os.path
from typing import List, Tuple, Union

import numpy as np
import piq
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as transforms
from numpy.typing import NDArray
from tqdm import tqdm

from fhibe_eval_api.common.loaders import image_data_loader_from_paths
from fhibe_eval_api.common.utils import check_list_lengths
from fhibe_eval_api.metrics.face_verification.curricular_face.curricular_face import (
    curricular_face_model,
)
from fhibe_eval_api.metrics.face_verification.utils import align_faces


@torch.no_grad()
def learned_perceptual_image_patch_similarity_score(
    image_paths_1: List[str],
    image_paths_2: List[str],
    exif_transpose: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
) -> List[float]:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS) between paired images.

    Args:
        image_paths_1 (List[str]): A list of strings representing the paths to the
            first set of image files.
        image_paths_2 (List[str]): A list of strings representing the paths to the
            second set of image files.
        exif_transpose (bool): A boolean indicating whether to apply EXIF orientation
            correction. Default is True.
        batch_size (int): The batch size to use when loading images. Default is 32.
        num_workers (int): Number of workers to load data in parallel.

    Returns:
        A list of floats representing the LPIPS between paired images.
    """
    check_list_lengths(list_1=image_paths_1, list_2=image_paths_2)

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    data_loader_1, data_loader_2 = image_data_loader_from_paths(
        image_paths_1=image_paths_1,
        image_paths_2=image_paths_2,
        transform=transform,
        exif_transpose=exif_transpose,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    lpips_metric = piq.LPIPS(reduction="none")

    lpips_scores = []

    with tqdm(
        total=len(image_paths_1),
        desc="Computing LPIPS between paired images",
    ) as pbar:
        for image_batch_1, image_batch_2 in zip(data_loader_1, data_loader_2):
            batch_lpips_scores = lpips_metric(
                image_batch_1["images"], image_batch_2["images"]
            )
            lpips_scores.extend(batch_lpips_scores)
            pbar.update(len(image_batch_1["images"]))

    return [lpips_score.item() for lpips_score in lpips_scores]


def peak_signal_to_noise_ratio_score(
    image_paths_1: List[str],
    image_paths_2: List[str],
    exif_transpose: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
) -> List[float]:
    """Calculate peak signal-to-noise ratio (PSNR) between paired images.

    Args:
        image_paths_1 (List[str]): A list of strings representing the paths to the
            first set of image files.
        image_paths_2 (List[str]): A list of strings representing the paths to the
            second set of image files.
        exif_transpose (bool): A boolean indicating whether to apply EXIF orientation
            correction. Default is True.
        batch_size (int): The batch size to use when loading images. Default is 32.
        num_workers (int): Number of workers to load data in parallel.

    Returns:
        A list of floats representing the PSNR between paired images.
    """
    check_list_lengths(list_1=image_paths_1, list_2=image_paths_2)

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    data_loader_1, data_loader_2 = image_data_loader_from_paths(
        image_paths_1=image_paths_1,
        image_paths_2=image_paths_2,
        transform=transform,
        exif_transpose=exif_transpose,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    psnr_scores = []
    with tqdm(
        total=len(image_paths_1),
        desc="Compute the Peak Signal-to-Noise Ratio (PSNR) between paired images",
    ) as pbar:
        for image_batch_1, image_batch_2 in zip(data_loader_1, data_loader_2):
            batch_psnr_scores = piq.psnr(
                image_batch_1["images"],
                image_batch_2["images"],
                data_range=1.0,
                reduction="none",
            )
            psnr_scores.extend(batch_psnr_scores)
            pbar.update(len(image_batch_1["images"]))

    return [psnr_score.item() for psnr_score in psnr_scores]


@torch.no_grad()
def curricular_face_score(
    image_paths_1: List[str],
    image_paths_2: List[str],
    exif_transpose: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    images_aligned: bool = True,
    cuda: bool = True,
) -> List[float]:  # pragma: no cover
    """Compute the CurricularFace embedding similarity between paired images.

    Args:
        image_paths_1: A list of strings representing the paths to the
            first set of image files.
        image_paths_2: A list of strings representing the paths to the
            second set of image files.
        exif_transpose: A boolean indicating whether to apply EXIF orientation
            correction. Default is True.
        batch_size: The batch size to use when loading images. Default is 32.
        num_workers: Number of workers to load data in parallel.
        images_aligned: Whether images_paths_1 and 2 are already aligned.
        cuda: Whether to use the GPU for running the CurricularFace model.

    Returns:
        A list of floats representing the CurricularFace embedding similarities
            between paired images.
    """
    check_list_lengths(list_1=image_paths_1, list_2=image_paths_2)

    if not images_aligned:
        aligned_img_filepaths_1 = []
        for x in image_paths_1:
            aligned_img_filepaths_1.append("temp/temp_aligned_1_" + os.path.basename(x))

        aligned_img_filepaths_2 = []
        for x in image_paths_2:
            aligned_img_filepaths_2.append("temp/temp_aligned_2_" + os.path.basename(x))

        success_1 = align_faces(
            img_filepaths=image_paths_1,
            aligned_img_filepaths=aligned_img_filepaths_1,
            crop_size=112,
            cuda=cuda,
        )
        success_2 = align_faces(
            img_filepaths=image_paths_2,
            aligned_img_filepaths=aligned_img_filepaths_2,
            crop_size=112,
            cuda=cuda,
        )
        if not (any(success_1)):
            raise RuntimeError("Unable to align encoded images")
        if not (any(success_2)):
            raise RuntimeError("Unable to align pre-aligned images")

        for x in range(len(success_1)):  # type: ignore
            if success_1[x] != success_2[x]:  # type: ignore
                success_1[x] = False  # type: ignore
                success_2[x] = False  # type: ignore
    else:
        success_1 = [True for _ in range(len(image_paths_1))]
        success_2 = success_1
        aligned_img_filepaths_1 = image_paths_1
        aligned_img_filepaths_2 = image_paths_2

    model = curricular_face_model()

    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    data_loader_1, data_loader_2 = image_data_loader_from_paths(
        image_paths_1=[x for x, s in zip(aligned_img_filepaths_1, success_1) if s],
        image_paths_2=[x for x, s in zip(aligned_img_filepaths_2, success_2) if s],
        transform=transform,
        exif_transpose=exif_transpose,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    image_embeddings_1 = []
    image_embeddings_2 = []
    with tqdm(
        total=len(image_paths_1),
        desc="Compute the CurricularFace Embedding Similarity between paired images",
    ) as pbar:
        for image_batch_1, image_batch_2 in zip(data_loader_1, data_loader_2):
            batch_embeddings_1, _ = model(image_batch_1["images"].cuda())
            batch_embeddings_2, _ = model(image_batch_2["images"].cuda())

            batch_embeddings_1 = batch_embeddings_1.cpu()
            batch_embeddings_2 = batch_embeddings_2.cpu()

            normed_batch_embeddings_1 = F.normalize(batch_embeddings_1, p=2, dim=1)
            normed_batch_embeddings_2 = F.normalize(batch_embeddings_2, p=2, dim=1)

            image_embeddings_1.append(normed_batch_embeddings_1)
            image_embeddings_2.append(normed_batch_embeddings_2)
            pbar.update(len(image_batch_1["images"]))

    torch_image_embeddings_1 = torch.cat(image_embeddings_1, dim=0)
    torch_image_embeddings_2 = torch.cat(image_embeddings_2, dim=0)

    dot_products = (torch_image_embeddings_1 * torch_image_embeddings_2).sum(dim=1)
    dot_products_list = [dot_product for dot_product in dot_products.tolist()]

    if not images_aligned:
        for x in aligned_img_filepaths_1 + aligned_img_filepaths_2:
            try:
                os.remove(x)
            except FileNotFoundError:
                pass

    if len(dot_products_list) == len(image_paths_1):
        return dot_products_list
    else:
        full_dot_products_list = []
        dot_prod_idx = 0
        for x in success_1:  # type: ignore
            if x:
                full_dot_products_list.append(dot_products_list[dot_prod_idx])
                dot_prod_idx += 1
            else:
                full_dot_products_list.append(None)
        return full_dot_products_list


def bbox_intersection_over_union_score(
    box_1: Tuple[float, float, float, float], box_2: Tuple[float, float, float, float]
) -> float:
    """Calculate the Intersection over Union (IOU) between two bounding boxes.

    Args:
        box_1 (tuple of floats): A tuple containing the coordinates of the first
            bounding box in the format (x_1, y_1, x_2, y_2) where (x_1, y_1) and (x_2,
            y_2) are the coordinates of the top-left and bottom-right corners of the
            bounding box.
        box_2 (tuple of floats): A tuple containing the coordinates of the second
            bounding box in the format (x_1, y_1, x_2, y_2) where (x_1, y_1) and (x_2,
            y_2) are the coordinates of the top-left and bottom-right corners of the
            bounding box.

    Returns:
        The IOU (float) between the two bounding boxes.
    """
    # Calculate the coordinates of the intersection rectangle
    x_left = max(box_1[0], box_2[0])
    y_top = max(box_1[1], box_2[1])
    x_right = min(box_1[2], box_2[2])
    y_bottom = min(box_1[3], box_2[3])

    # If the intersection is empty, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    box_1_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    box_2_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    # Calculate the IOU by dividing the intersection area by the union area
    iou = intersection_area / float(box_1_area + box_2_area - intersection_area)

    return iou


def segmentation_intersection_over_union_score(
    pred_mask: NDArray[np.uint8], gt_mask: NDArray[np.uint8]
) -> float:
    """Computes the intersection over union (IoU) for two binary segmentation masks.

    Args:
        pred_mask: NumPy array of shape (height, width) representing the
            predicted segmentation mask.
        gt_mask: NumPy array of shape (height, width) representing the
            ground-truth segmentation mask.

    Returns:
        iou (float): The IoU score between the two masks.
    """
    # Ensure that both masks are binary
    # the input is np.uint8; np.positive keeps the type.
    pred_mask = np.positive(pred_mask)
    gt_mask = np.positive(gt_mask)

    # TODO: delete -- the code is equivalent.
    # Calculate the intersection and union
    # intersection = np.logical_and(gt_mask, pred_mask).sum()
    # union = np.logical_or(gt_mask, pred_mask).sum()

    intersection = np.count_nonzero(gt_mask * pred_mask)
    union = np.count_nonzero(np.maximum(gt_mask, pred_mask))

    # Calculate the IoU score
    iou = intersection / union if union > 0 else 0

    return iou


def f1_score(pred_mask: NDArray[np.uint8], gt_mask: NDArray[np.uint8]) -> float:
    """Compute the F1 score for segmentation masks.

    Args:
        pred_mask: NumPy array of shape (height, width) representing the
            predicted segmentation mask.
        gt_mask: NumPy array of shape (height, width) representing the
            ground-truth segmentation mask.

    Returns:
        f1_score (float): The F1 score between the two masks.
    """
    # Ensure that both masks are binary
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    # True Positives (TP): Intersection of predicted and ground truth masks
    tp = np.count_nonzero(np.logical_and(pred_mask, gt_mask))

    # False Positives (FP): Predicted mask minus true positives
    fp = np.count_nonzero(np.logical_and(pred_mask, np.logical_not(gt_mask)))

    # False Negatives (FN): Ground truth mask minus true positives
    fn = np.count_nonzero(np.logical_and(np.logical_not(pred_mask), gt_mask))

    # F1 score
    f1_value = 2 * tp / (2 * tp + fp + fn)
    return f1_value


def best_iou_scores_for_gt_boxes(
    pred_boxes: list[tuple[float, float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]],
) -> list[float]:
    """For every ground truth box, returns the highest IoU with a predicted box."""
    best_iou_scores = []

    for gt_box in gt_boxes:
        best_iou = 0.0
        for pred_box in pred_boxes:
            iou = bbox_intersection_over_union_score(pred_box[:4], gt_box)
            best_iou = max(best_iou, iou)
        best_iou_scores.append(best_iou)

    return best_iou_scores


def best_iou_scores_for_gt_masks(
    pred_masks: list[NDArray[np.uint8]],
    gt_masks: list[NDArray[np.uint8]],
) -> list[float]:
    """Get the best iou for each ground truth mask.

    Args:
        pred_masks: A list of predicted masks,
            each represented as a np.ndarray.
        gt_masks: A list of ground-truth masks,
            each represented as a np.ndarray.

    Returns:
        List of best iou score for each GT mask.
    """
    best_iou_scores = []
    for gt_mask in gt_masks:
        best_iou = 0.0
        for pred_mask in pred_masks:
            iou = segmentation_intersection_over_union_score(pred_mask, gt_mask)
            best_iou = max(best_iou, iou)
        best_iou_scores.append(best_iou)
    return best_iou_scores


def percentage_correct_keypoints_score(
    pred_keypoints: NDArray[np.uint8],
    gt_keypoints: NDArray[np.uint8],
    face_bbox: Tuple[int, int, int, int],
    thresholds: List[float],
    visible: NDArray[np.uint8],
) -> Union[List[float], float]:
    """Compute the Percentage of Correct Keypoints (PCK).

    Given the predicted and ground-truth keypoints for a single image
    and a list of thresholds.

    Args:
        pred_keypoints: A NumPy array of shape (num_keypoints, 2)
            containing the predicted keypoints.
        gt_keypoints: A NumPy array of shape (num_keypoints, 2)
            containing the ground-truth keypoints.
        face_bbox (tuple of ints): A tuple (x_1, y_1, x_2, y_2) representing the
            bounding box of the face in the image.
        thresholds (list of floats): A list of float values representing the PCK
            thresholds.
        visible: A NumPy array of shape (num_keypoints, ) indicating
            which keypoints are visible in the ground truth.

    Returns:
        pck_scores (list of floats or float): A list of PCK scores for each
            threshold value in `thresholds`.
    """
    if not any(visible):
        return None
    # Euclidean distance between the predicted keypoints and ground-truth keypoints
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)

    # Threshold distance based on the face bounding box's diagonal length
    diagonal_length = np.sqrt(
        (face_bbox[2] - face_bbox[0]) ** 2 + (face_bbox[3] - face_bbox[1]) ** 2
    )
    threshold_distances = [threshold * diagonal_length for threshold in thresholds]

    # Number of keypoints that are correctly predicted within each threshold distance
    num_correct = [
        np.sum(distances[visible] <= threshold) for threshold in threshold_distances
    ]
    # Compute the PCK scores for each threshold value
    pck_scores: List[float] = [num / np.sum(visible) for num in num_correct]

    if len(pck_scores) == 1:
        return pck_scores[0]
    else:
        return pck_scores


def best_oks_scores_for_gt_keypoints(
    pred_keypoints: List[NDArray[np.uint8]],
    gt_keypoints: List[NDArray[np.uint8]],
    gt_persona_segment_areas: List[int],
    kpt_oks_sigmas: NDArray[np.uint8],
) -> list[float]:
    """Compute highest IoU(OKS) for each ground truth keypoint set."""
    best_iou_scores = []

    for gt, gt_persona_segment_area in zip(gt_keypoints, gt_persona_segment_areas):
        best_iou = None
        for pred in pred_keypoints:
            iou = object_keypoint_similarity(
                dt=pred,
                gt=gt,
                gt_persona_segment_area=gt_persona_segment_area,
                kpt_oks_sigmas=kpt_oks_sigmas,
            )
            if best_iou is None:
                best_iou = iou
            else:
                best_iou = max(best_iou, iou)
        best_iou_scores.append(best_iou)

    return best_iou_scores


def object_keypoint_similarity(
    dt: NDArray[np.uint8],
    gt: NDArray[np.uint8],
    gt_persona_segment_area: int,
    kpt_oks_sigmas: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Compute the Object Keypoint Similarity (OKS).

    Adapted from from pycocotools.
    Given the predicted and ground-truth keypoints for a single image
    and a list of thresholds.
    OKS plays the same role in pose estimation as the IoU in detection/pasring tasks.

    Args:
        dt (list of np.ndarray): A NumPy array of shape (num_keypoints, 2)
            containing the predicted keypoints.
        gt (list of np.ndarray): A NumPy array of shape (num_keypoints, 3)
            containing the ground-truth keypoints and visibility.
        gt_persona_segment_area (int): the pixels this subject takes
            (not the bbox but segment).
        kpt_oks_sigmas: Per-keypoint standard deviation, used to scale
            the distance of different keypoints: some keypoints has higher varaince
            of annotations than others, like shoulders have high variance than eyes.

    Returns:
        Object keypoint similarities for each ground truth keypoint set.
    """
    sigmas = kpt_oks_sigmas

    vars = (sigmas * 2) ** 2

    # compute oks between each detection and ground truth object
    g = np.array(gt)
    xg = np.array([kpt[0] for kpt in g])
    yg = np.array([kpt[1] for kpt in g])
    vg = np.array([kpt[2] for kpt in g])
    k1 = np.count_nonzero(vg > 0)
    if k1 == 0:
        # no keypoints visible
        return None

    # compute iou for prediction
    d = np.array(dt)
    xd = np.array([kpt[0] for kpt in d])
    yd = np.array([kpt[1] for kpt in d])
    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    e = (dx**2 + dy**2) / vars / (gt_persona_segment_area + np.spacing(1)) / 2
    if k1 > 0:
        e = e[vg > 0]
    iou = np.sum(np.exp(-e)) / e.shape[0]
    return iou
