# Copyright (c) Sony AI Inc.
# All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torchvision.transforms as transforms
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fhibe_eval_api.common.utils import get_project_root
from fhibe_eval_api.evaluate import evaluate_task
from fhibe_eval_api.models.base_model import BaseModelWrapper
from fhibe_eval_api.reporting import BiasReport

project_root = get_project_root()
batch_size = 4


class FaceDataSet(Dataset):
    """Reusable face dataset for the face parsing task."""

    def __init__(  # noqa: D107
        self, img_filepaths: List[str], crop_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """Constructor.

        Args:
            img_filepaths: List of image filepaths.
            crop_size: A list of two intergers.

        Return:
            None
        """
        if crop_size is None:
            crop_size = (473, 473)
        self.img_filepaths = img_filepaths
        self.crop_size = np.asarray(crop_size)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform = transform

        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]

        self.number_samples = len(self.img_filepaths)

    def __len__(self) -> int:  # noqa: D105
        return self.number_samples

    def _box2cs(self, box: List[float]) -> Tuple[Any]:
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x: float, y: float, w: float, h: float) -> Tuple[Any]:
        """Convert x,y,width,height of an image to center, scale.

        Args:
            x: minimum x coordinate of image
            y: minimum y coordinate of image
            w: width of image
            h: height of image

        Return:
            Tuple containing the center and scale of the image.
        """
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, idx: int) -> tuple:  # noqa: D105
        img_filepath = self.img_filepaths[idx]
        im = cv2.imread(img_filepath, cv2.IMREAD_COLOR)
        h, w, _ = im.shape

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        trans = get_affine_transform(center, s, r, self.crop_size)
        image = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        if self.transform:
            image = self.transform(image)

        meta = {
            "name": img_filepath,
            "center": center,
            "height": h,
            "width": w,
            "scale": s,
            "rotation": r,
            "origin": image,
        }
        return image, meta


def get_affine_transform(  # noqa: D103
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    if not isinstance(shift, np.ndarray):
        shift = np.array(shift)

    if not isinstance(scale, np.ndarray):
        scale = np.array(scale)

    if not isinstance(center, np.ndarray):
        center = np.array(center)

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    if not isinstance(src_dir, np.ndarray):
        src_dir = np.array(src_dir)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):  # noqa: D103
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):  # noqa: D103
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def data_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Get a data loader for the task.

    Args:
        dataset: Face dataset object
        batch_size: # of images in each batch
        shuffle: Whether to shuffle batches
        pin_memory: Whether to pin memory.

    Return:
        Torch dataloader.
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )


class CustomModel(nn.Module):
    """A dummy model illustrating an example custom model for this task."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.batch_size = batch_size

    def forward(self, batch) -> List[Dict[str, List[Any]]]:
        """Perform a forward pass (inference) of a batch of data.

        Args:
            batch: A batch from a data loader

        Return:
            List of dicts, where each dict contains the parsed masks.
        """
        images, meta = batch
        results = []
        for i in range(len(images)):
            # CelebAMask-HQ has 19 integer classes which we encode as 0-18
            dets = np.random.randint(
                0, 19, size=(images.shape[2], images.shape[3]), dtype=np.uint8
            )
            results.append(
                {
                    "detections": dets,
                }
            )
        return results


class DemoFaceParser(BaseModelWrapper):
    """Model wrapper to comply with API standards."""

    def __init__(self, model: Any, map_ears_to_skin: bool):
        """Initialize the object by referencing the base class.

        Args:
            model: An instance of your custom model class.
            map_ears_to_skin: Whether to map ear predictions to
                face skin predictions since FHIBE considers ears
                part of the face skin.

        Return:
            None
        """
        super().__init__(model)
        self.map_ears_to_skin = map_ears_to_skin
        self.input_size = (512, 512)

    def data_preprocessor(
        self, img_filepaths: List[str], **kwargs: Dict[str, Any]
    ) -> DataLoader:
        """Perform batch preprocessing and return a data loader.

        Args:
            img_filepaths: List of unique image filepaths
            **kwargs: additional keyword arguments.

        Return:
            Torch dataloader
        """
        dataset = FaceDataSet(img_filepaths=img_filepaths, crop_size=(473, 473))

        dataset_loader = data_loader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        height, width = self.input_size
        self.interp = nn.Upsample(
            size=(height, width), mode="bilinear", align_corners=True
        )

        return dataset_loader

    def __call__(self, batch: Any) -> List[Dict[str, List[Any]]]:
        """Run forward pass over the demo model.

        Args:
            batch: a batch of data from the data loader

        Return:
            List of dicts, where each dict contains the parsed masks.
        """
        return self.model(batch)


def main() -> None:
    """Run the demo."""
    np.random.seed(0)
    with open("./face_parsing.yaml", "r") as f:
        config = yaml.safe_load(f)

    task_name = config["task_name"]
    dataset_name = config["dataset_name"]
    data_rootdir = config["data_rootdir"]
    model_name = config["model_name"]
    metrics = config["metrics"]
    attributes = config["attributes"]
    downsampled = config["downsampled"]
    use_mini_dataset = config["use_mini_dataset"]
    dataset_version = config["dataset_version"]
    results_basedir = config["results_basedir"]

    face_parsing_model = CustomModel()
    wrapped_model = DemoFaceParser(face_parsing_model, map_ears_to_skin=True)

    evaluate_task(
        task_name=task_name,
        dataset_name=dataset_name,
        data_rootdir=data_rootdir,
        model=wrapped_model,
        model_name=model_name,
        metrics=metrics,
        attributes=attributes,
        results_rootdir=results_basedir,
        use_mini_dataset=use_mini_dataset,
        downsampled=downsampled,
    )
    bias_report = BiasReport(
        model_name=model_name,
        task_name=task_name,
        data_rootdir=data_rootdir,
        dataset_version=dataset_version,
        results_base_dir=results_basedir,
        dataset_name=dataset_name,
        downsampled=downsampled,
        use_mini_dataset=use_mini_dataset,
    )
    bias_report.generate_pdf_report(
        attributes=attributes,
    )


if __name__ == "__main__":
    main()
