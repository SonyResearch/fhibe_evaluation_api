# SPDX-License-Identifier: Apache-2.0
"""Module implementing the MTCNN network."""

import math
import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from facenet_pytorch import MTCNN as facenet_MTCNN  # noqa: N811
from PIL import Image
from torch.autograd import Variable

# SPDX-License-Identifier: Apache-2.0


CURRENT_DIR = os.path.dirname(__file__)


class Flatten(nn.Module):
    def __init__(self):
        """Flattens a 4D tensor of shape.

        Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
        applications/align
        """
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be flattened.

        Returns:
            torch.Tensor: Flattened tensor.
        """
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class PNet(nn.Module):
    def __init__(self):
        """Initializes the P-Net.

        Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
        applications/align
        """
        super(PNet, self).__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 10, 3, 1)),
                    ("prelu1", nn.PReLU(10)),
                    ("pool1", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(10, 16, 3, 1)),
                    ("prelu2", nn.PReLU(16)),
                    ("conv3", nn.Conv2d(16, 32, 3, 1)),
                    ("prelu3", nn.PReLU(32)),
                ]
            )
        )

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        model_weights_filepath = os.path.join(CURRENT_DIR, "pnet.npy")
        if not os.path.isfile(model_weights_filepath):
            raise FileNotFoundError(f"Could not find {model_weights_filepath}")
        weights = np.load(model_weights_filepath, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the P-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors b and a.
                - b (torch.Tensor): Output tensor of shape (batch_size, 4, H', W').
                - a (torch.Tensor): Output tensor of shape (batch_size, 2, H', W').
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a)
        return b, a


class RNet(nn.Module):
    def __init__(self):
        """Initializes the R-Net.

        Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
        applications/align
        """
        super(RNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 28, 3, 1)),
                    ("prelu1", nn.PReLU(28)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(28, 48, 3, 1)),
                    ("prelu2", nn.PReLU(48)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(48, 64, 2, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("flatten", Flatten()),
                    ("conv4", nn.Linear(576, 128)),
                    ("prelu4", nn.PReLU(128)),
                ]
            )
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        model_weights_filepath = os.path.join(CURRENT_DIR, "rnet.npy")
        if not os.path.isfile(model_weights_filepath):
            raise FileNotFoundError(f"Could not find " f"{model_weights_filepath}")

        weights = np.load(model_weights_filepath, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the R-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, h, w).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors b and a.
                - b (torch.Tensor): Output tensor of shape (batch_size, 4).
                - a (torch.Tensor): Output tensor of shape (batch_size, 2).
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a)
        return b, a


class ONet(nn.Module):
    def __init__(self):
        """Initializes the O-Net.

        Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
        applications/align
        """
        super(ONet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1)),
                    ("prelu1", nn.PReLU(32)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("prelu2", nn.PReLU(64)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(64, 64, 3, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv4", nn.Conv2d(64, 128, 2, 1)),
                    ("prelu4", nn.PReLU(128)),
                    ("flatten", Flatten()),
                    ("conv5", nn.Linear(1152, 256)),
                    ("drop5", nn.Dropout(0.25)),
                    ("prelu5", nn.PReLU(256)),
                ]
            )
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        model_weights_filepath = os.path.join(CURRENT_DIR, "onet.npy")
        if not os.path.isfile(model_weights_filepath):
            raise FileNotFoundError(f"Could not find " f"{model_weights_filepath}")

        weights = np.load(model_weights_filepath, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the O-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, h, w).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensors c, b and a.
                - c (torch.Tensor): Output tensor of shape (batch_size, 10).
                - b (torch.Tensor): Output tensor of shape (batch_size, 4).
                - a (torch.Tensor): Output tensor of shape (batch_size, 2).
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a)
        return c, b, a


class face_evolve_MTCNN:  # noqa: N801
    def __init__(
        self,
        min_face_size: float = 20.0,
        thresholds: Optional[Tuple[float, float, float]] = None,
        nms_thresholds: Optional[Tuple[float, float, float]] = None,
        device: str = "cuda",
    ):
        """MTCNN face detector.

        Args:
            image_size (int): The size of the input image. Default is 160.
            min_face_size (int): The minimum face size to detect. Default is 20.
            thresholds (Tuple[float, float, float]): List of length 3, containing the
                detection thresholds for each stage of the face detection pipeline.
            nms_thresholds (Tuple[float, float, float]): List of length 3, containing
                the NMS thresholds for each stage of the face detection pipeline.
            device (str): Device to use. Default is 'cuda'.
        """
        super(face_evolve_MTCNN, self).__init__()

        self.min_face_size = min_face_size

        if nms_thresholds is None:
            nms_thresholds = (0.7, 0.7, 0.7)
        if thresholds is None:
            thresholds = (0.6, 0.7, 0.8)

        self.nms_thresholds = nms_thresholds
        self.thresholds = thresholds

        # LOAD MODELS
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet().eval()

        if device == "cuda":
            self.pnet = self.pnet.cuda()
            self.rnet = self.rnet.cuda()
            self.onet = self.onet.cuda()

    def detect_faces(
        self, image: Image.Image
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects faces in an image and returns bounding boxes and facial landmarks.

        Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
        applications/align

        Args:
            image (Image.Image): PIL image object.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Output arrays
                bounding_boxes and landmarks. Returns Tuple[None, None] if no face
                is detected.
                - bounding_boxes (np.ndarray): Bounding boxes for all the detected
                    faces. It has shape [n_boxes, 5], where each row contains the
                    coordinates of a bounding box in the format (x1, y1, x2, y2,
                    score), where (x1, y1) are the coordinates of the top-left corner
                    of the box, (x2, y2) are the coordinates of the bottom-right
                    corner, and 'score' is the confidence score of the detection.
                    Returns None if no face is detected.
                - landmarks (np.ndarray): Facial landmarks for all the detected faces.
                    It has shape [n_boxes, 10], where each row contains the
                    coordinates of the 5 facial landmarks for a face in the format
                    (x1, y1, x2, y2, ..., x5, y5), where (xi, yi) are the coordinates
                    of the ith landmark point. Returns None is no face if detected.
        """
        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes_list = []

        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(
                image, self.pnet, scale=s, threshold=self.thresholds[0]
            )
            bounding_boxes_list.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes_tuple: Tuple[np.ndarray, ...] = tuple(
            [i for i in bounding_boxes_list if i is not None]
        )
        # if len(bounding_boxes_tuple) == 0 or None in bounding_boxes_tuple:
        if len(bounding_boxes_tuple) == 0 or any(
            x is None for x in bounding_boxes_tuple
        ):
            # raise ValueError("bounding box tuple list contains None or is empty.")
            return None, None

        bounding_boxes = np.vstack(bounding_boxes_tuple)

        keep = nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0].tolist()
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, self.nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return None, None

        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0].tolist()
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = (
            np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        )
        landmarks[:, 5:10] = (
            np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        )

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, self.nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        return bounding_boxes, landmarks


def run_first_stage(
    image: Image.Image, net: nn.Module, scale: float, threshold: float
) -> Optional[np.ndarray]:
    """Run P-Net, generate bounding boxes, and perform non-maximum suppression (NMS).

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        image (Image.Image): PIL image object.
        net (nn.Module): P-Net module.
        scale (float): Value to scale the width and height of the image by.
        threshold (float): Threshold on the probability of a face when generating
            bounding boxes from predictions.

    Returns:
        A float numpy array of shape (n_boxes, 9), representing bounding boxes with
            scores and offsets (4 + 1 + 4). Returns None if no bounding boxes are found.
    """
    # scale the image and convert it to a float array
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, "float32")

    img = Variable(torch.FloatTensor(_preprocess(img)), volatile=True)
    output = net(img)
    probs = output[1].data.numpy()[0, 1, :, :]
    offsets = output[0].data.numpy()
    # probs: probability of a face at each sliding window
    # offsets: transformations to true bounding boxes

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(
    probs: np.ndarray, offsets: np.ndarray, scale: float, threshold: float
) -> np.ndarray:
    """Generate bounding boxes at places where there is probably a face.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        probs (np.ndarray): A float numpy array of shape (n, m) containing
            probabilities.
        offsets (np.ndarray): A float numpy array of shape (1, 4, n, m) containing
            offsets.
        scale (float): Value to scale the width and height of the image by.
        threshold (float): Threshold on the probability of a face when generating
            bounding boxes from predictions.

    Returns:
        np.ndarray: A float numpy array of shape (n_boxes, 9) containing bounding
            boxes.
    """
    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack(
        [
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score,
            offsets,
        ]
    )
    # why one is added?
    return bounding_boxes.T


def nms(
    boxes: np.ndarray, overlap_threshold: float = 0.5, mode: str = "union"
) -> List[int]:
    """Performs non-maximum suppression on a set of bounding boxes.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        boxes (np.ndarray): A float numpy array of shape (n, 5), where each row is
            (xmin, ymin, xmax, ymax, score).
        overlap_threshold (float): Float representing the IoU threshold above which
            boxes are suppressed.
        mode (str): Either 'union' (IoU) or 'min' (minimum overlap).

    Returns:
        A list of indices representing the boxes that are selected after NMS.

    Raises:
        NotImplementedError: If mode is not 'union' or 'min'.
    """
    if mode not in ["min", "union"]:
        raise NotImplementedError(f"mode: {mode} is not implemented")

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:
        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        else:
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def convert_to_square(bboxes: np.ndarray) -> np.ndarray:
    """Converts bounding boxes to a square form.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        bboxes (np.ndarray): Float numpy array of shape (n, 5). Each row represents a
            bounding box with coordinates (x1, y1, x2, y2) and a confidence score.

    Returns:
        np.ndarray: Float numpy array of shape (n, 5) representing squared bounding
            boxes.
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Transform bounding boxes to be more like true bounding boxes.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        bboxes: Float numpy array of shape (n, 5).
        offsets: Float numpy array of shape (n, 4).

    Returns:
        np.ndarray: Float numpy array of shape (n, 5).
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(
    bounding_boxes: np.ndarray, img: Image.Image, size: int = 24
) -> np.ndarray:
    """Cut out boxes from the image.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        bounding_boxes (np.ndarray): Float numpy array of shape (n, 5).
        img (Image.Image): PIL image object.
        size (int): Size of cutouts.

    Returns:
        np.ndarray: Float numpy array of shape (n, 3, size, size).
    """
    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
        bounding_boxes, width, height
    )
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")

        img_array = np.asarray(img, "uint8")
        img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[
            y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :
        ]

        # resize
        pil_img_box = Image.fromarray(img_box)
        pil_img_box = pil_img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(pil_img_box, "float32")

        img_boxes[i, :, :, :] = _preprocess(img_box)
    return img_boxes


def correct_bboxes(bboxes: np.ndarray, width: float, height: float) -> List[np.ndarray]:
    """Crop boxes that are too big and get coordinates with respect to cutouts.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        bboxes (np.ndarray): Float numpy array of shape (n, 5), where each row is
            (xmin, ymin, xmax, ymax, score).
        width (float): Width.
        height (float): Height.

    Returns:
        List[np.ndarray] in the following order: [dy, edy, dx, edx, y, ey, x, ex, w, h].
            - dy, dx, edy, edx: Integer numpy arrays of shape (n), coordinates of the
                boxes with respect to the cutouts.
            - y, x, ey, ex: Integer numpy arrays of shape (n), corrected ymin, xmin,
                ymax, xmax.
            - h, w: Integer numpy arrays of shape(n), just heights and widths of boxes.
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype("int32") for i in return_list]

    return return_list


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocessing step before feeding the network.

    Adapted from: https://github.com/ZhaoJ9014/face.evoLVe/blob/master/
    applications/align

    Args:
        img (np.ndarray): a float numpy array of shape (h, w, c).

    Returns:
        np.ndarray: Float numpy array of shape (1, c, h, w).
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def mtcnn_model(
    implementation: str = "facenet",
    image_size: int = 160,
    margin: int = 0,
    min_face_size: int = 20,
    thresholds: Optional[Tuple[float, float, float]] = None,
    nms_thresholds: Optional[Tuple[float, float, float]] = None,
    factor: float = 0.709,
    post_process: bool = True,
    select_largest: bool = False,
    selection_method: Optional[str] = "probability",
    keep_all: bool = True,
    cuda: bool = True,
) -> Union[nn.Module, face_evolve_MTCNN]:
    """Creates an instance of MTCNN face detection model.

    Args:
        implementation (str): Options are 'face_evolve' or 'facenet'.
        image_size (int): The size of the output image. Default is 160.
        margin (int): Margin added to the bounding box detected around the face.
            Default is 0.
        min_face_size (int): The minimum face size to detect. Default is 20.
        thresholds (tuple, optional): A list of three float thresholds for the three
            stages of the MTCNN model. Default is [0.6, 0.7, 0.7].
        nms_thresholds (tuple, optional): List of length 3, containing the NMS
            thresholds for each stage of the face detection pipeline.
        factor (float): The scale factor used to create a pyramid of images for the
            MTCNN model. Default is 0.709.
        post_process (bool): Whether to post-process the detected bounding boxes.
            Default is True.
        select_largest (bool): Whether to select only the largest bounding box if
            multiple boxes are detected. Default is False.
        selection_method (str, optional): The method used to select the bounding box
            from multiple boxes detected. Either 'largest' or 'probability'. Default
            is 'probability'.
        keep_all (bool): Whether to keep all the detected bounding boxes. Default
            is True.
        cuda (bool): If True model is moved to cuda device.

    Returns:
        An instance of the MTCNN face detection model.

    Raises:
        ValueError: If selection_method is not 'largest' or 'probability'.
    """
    implementation = implementation.lower()
    if implementation not in ["face_evolve", "facenet"]:
        raise ValueError(
            f"implementation must be 'face_evolve' or 'facenet' not "
            f"{implementation}"
        )

    if implementation == "facenet":
        if thresholds is None:
            # thresholds = (0.6, 0.7, 0.7)
            thresholds = (0.6, 0.7, 0.02)

        return facenet_MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            select_largest=select_largest,
            selection_method=selection_method,
            keep_all=keep_all,
            device="cuda" if cuda else "cpu",
        ).eval()
    else:
        if thresholds is None:
            # thresholds = (0.6, 0.7, 0.8)
            thresholds = (0.6, 0.7, 0.02)

        if nms_thresholds is None:
            nms_thresholds = (0.7, 0.7, 0.7)

        return face_evolve_MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds,
            nms_thresholds=nms_thresholds,
            device="cuda" if cuda else "cpu",
        )
