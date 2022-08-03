# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import copy
# import dataclasses
# import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
# from packaging import version

from transformers.utils import (
    #TensorType,
    is_torch_available,
    is_vision_available,
    logging
)
# from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size


if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# if is_vision_available():
#     from PIL import Image

logger = logging.get_logger(__name__)


class CoreMLConfig(ABC):
    """
    Base class for Core ML exportable model describing metadata on how to export the model through the Core ML format.
    """

#     default_fixed_batch = 2
#     default_fixed_sequence = 8
#     default_fixed_num_choices = 4
#     torch_onnx_minimum_version = version.parse("1.8")
#     _tasks_to_common_outputs = {
#         "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
#         "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
#         "image-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
#         "image-segmentation": OrderedDict(
#             {
#                 "logits": {0: "batch", 1: "sequence"},
#                 "pred_boxes": {0: "batch", 1: "sequence"},
#                 "pred_masks": {0: "batch", 1: "sequence"},
#             }
#         ),
#         "masked-im": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
#         "masked-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
#         "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
#         "object-detection": OrderedDict(
#             {
#                 "logits": {0: "batch", 1: "sequence"},
#                 "pred_boxes": {0: "batch", 1: "sequence"},
#             }
#         ),
#         "question-answering": OrderedDict(
#             {
#                 "start_logits": {0: "batch", 1: "sequence"},
#                 "end_logits": {0: "batch", 1: "sequence"},
#             }
#         ),
#         "seq2seq-lm": OrderedDict({"logits": {0: "batch", 1: "decoder_sequence"}}),
#         "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
#         "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
#     }

    def __init__(self, config: "PretrainedConfig", task: str = "default"):
        self._config = config

        # if task not in self._tasks_to_common_outputs:
        #     raise ValueError(
        #         f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
        #     )
        self.task = task

    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "CoreMLConfig":
        """
        Instantiate a CoreMLConfig for a specific model

        Args:
            config: The model's configuration to use when exporting to Core ML

        Returns:
            CoreMLConfig for this model
        """
        return cls(config, task=task)

    @property
    def inputs(self) -> OrderedDict[str, Mapping[str, Any]]:
        """
        Ordered mapping of the inputs in the model

        Override this function to change the name of the inputs, their description strings,
        or any of the additional configuration options.

        Note: You are not allowed to change the order of the inputs!

        Image inputs can have the following options:

        - `"color_layout"`: `"RGB"` or `"BGR"` channel ordering
        - `"image_width"` and `"image_height"`: override the expected image size
        """
        if self.task == "image-classification":
            return OrderedDict(
                [
                    (
                        "image",
                        {
                            "description": "Image to be classified",
                            "color_layout": "RGB",
                        }
                    ),
                ]
            )

        if self.task == "masked-im":
            return OrderedDict(
                [
                    (
                        "image",
                        {
                            "description": "Input image",
                            "color_layout": "RGB",
                        }
                    ),
                    (
                        "bool_masked_pos",
                        {
                            "description": "Indicates which patches are masked (1) and which aren't (0)"
                        }
                    ),
                ]
            )

        raise AssertionError("Unsupported task '{self.task}'")

#     @property
#     def outputs(self) -> Mapping[str, Mapping[int, str]]:
#         """
#         Mapping containing the axis definition of the output tensors to provide to the model

#         Returns:
#             For each output: its name associated to the axes symbolic name and the axis position within the tensor
#         """
#         common_outputs = self._tasks_to_common_outputs[self.task]
#         return copy.deepcopy(common_outputs)

#     @property
#     def values_override(self) -> Optional[Mapping[str, Any]]:
#         """
#         Dictionary of keys to override in the model's config before exporting

#         Returns:
#             Dictionary with the keys (and their corresponding values) to override
#         """
#         if hasattr(self._config, "use_cache"):
#             return {"use_cache": False}

#         return None

#     @property
#     def default_batch_size(self) -> int:
#         """
#         The default batch size to use if no other indication

#         Returns:
#             Integer > 0
#         """
#         # Using 2 avoid ONNX making assumption about single sample batch
#         return OnnxConfig.default_fixed_batch

#     @property
#     def default_sequence_length(self) -> int:
#         """
#         The default sequence length to use if no other indication

#         Returns:
#             Integer > 0
#         """
#         return OnnxConfig.default_fixed_sequence

#     @property
#     def default_num_choices(self) -> int:
#         """
#         The default number of choices to use if no other indication

#         Returns:
#             Integer > 0
#         """
#         return OnnxConfig.default_fixed_num_choices

#     @property
#     def atol_for_validation(self) -> float:
#         """
#         What absolute tolerance value to use during model conversion validation.

#         Returns:
#             Float absolute tolerance value.
#         """
#         return 1e-5

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
    ) -> Mapping[str, np.ndarray]:
        """
        Generate inputs to provide to the Core ML exporter

        Args:
            preprocessor: ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.

        Returns:
            `Mapping[str, np.ndarray]` holding the tensors to provide to the model's forward function
        """
        from transformers.feature_extraction_utils import FeatureExtractionMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        input_names = list(self.inputs.keys())
        dummy_inputs = {}

        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # TODO: implement for text-based models
            pass

        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            if self.task in ["image-classification", "masked-im"]:
                if isinstance(preprocessor.size, tuple):
                    image_width, image_height = preprocessor.size
                else:
                    image_width = image_height = preprocessor.size

                pixel_values = np.random.rand(1, 3, image_height, image_width).astype(np.float32) * 2.0 - 1.0
                dummy_inputs[input_names.pop(0)] = pixel_values

            if self.task == "masked-im":
                num_patches = (self._config.image_size // self._config.patch_size) ** 2
                bool_masked_pos = np.random.randint(low=0, high=2, size=(1, num_patches)).astype(bool)
                dummy_inputs[input_names.pop(0)] = bool_masked_pos

        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

        return dummy_inputs

#     @classmethod
#     def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> Dict[str, Any]:
#         """
#         Flatten any potential nested structure expanding the name of the field with the index of the element within the
#         structure.

#         Args:
#             name: The name of the nested structure
#             field: The structure to, potentially, be flattened

#         Returns:
#             (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.

#         """
#         from itertools import chain

#         return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}
