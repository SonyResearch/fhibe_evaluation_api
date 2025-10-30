# SPDX-License-Identifier: Apache-2.0
"""Module containing the base model class used in evaluation.

All models used in the API must be wrapped with the base model wrapper.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModelWrapper(ABC):
    """A base model that wraps all custom models."""

    def __init__(self, model: Any):
        """Assign the custom model as an instance variable.

        Args:
            model: A custom model object, e.g., torch.nn.Module
        """
        self.model = model

    @abstractmethod
    def data_preprocessor(
        self, img_filepaths: List[str], **kwargs: Dict[str, Any]
    ) -> Any:
        """Perform any necessary preprocessing and return a data loader.

        Args:
            img_filepaths: A list of filepaths to use to build the data loader.
            kwargs: Additional keyword arguments.

        Return:
            A batch iterator, e.g., torch.utils.data.DataLoader.
        """
        pass  # pragma: no cover

    @abstractmethod
    def __call__(self, batch: Any) -> Any:
        """Perform a forward pass of a batch through the model.

        Args:
            batch: A batch of image data and possibly metadata,
                coming from a step in the data loader returned by
                data_preprocessor()

        Return:
            Task-specific. Typically, a list of dictionaries
            containing the model outputs for each image in the batch.
        """
        pass  # pragma: no cover
