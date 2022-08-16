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

        - `"color_layout"`: `"RGB"` or `"BGR"` channel ordering.
        """
        # TODO: the input for the default task depends on whether this is an image model or not

        if self.task in [
            "defaultX",
            "masked-lm",
            "question-answering",
            "sequence-classification",
            "token-classification",
        ]:
            return OrderedDict(
                [
                    (
                        "input_ids",
                        {
                            "description": "Indices of input sequence tokens in the vocabulary",
                            "sequence_length": 128,
                        }
                    ),
                    (
                        "attention_mask",
                        {
                            "description": "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        }
                    ),
                ]
            )

        if self.task in [
            "multiple-choice",
            "next-sentence-prediction",
        ]:
            return OrderedDict(
                [
                    (
                        "input_ids",
                        {
                            "description": "Indices of input sequence tokens in the vocabulary",
                            "sequence_length": 128,
                        }
                    ),
                    (
                        "attention_mask",
                        {
                            "description": "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        }
                    ),
                    (
                        "token_type_ids",
                        {
                            "description": "Segment token indices to indicate first and second portions of the inputs (0 = sentence A, 1 = sentence B)",
                        }
                    ),
                ]
            )

        if self.task in ["default", "object-detection", "semantic-segmentation"]:
            return OrderedDict(
                [
                    (
                        "image",
                        {
                            "description": "Input image",
                            "color_layout": "RGB",
                        }
                    ),
                ]
            )

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

    @property
    def outputs(self) -> OrderedDict[str, Mapping[int, str]]:
        """
        Ordered mapping of the outputs in the model

        Override this function to change the name of the outputs, their description strings,
        or any of the additional configuration options.

        Note: You are not allowed to change the order of the outputs!

        Semantic segmentation outputs can have the following options:

        - `do_upsample`: Scales the output to have the same width and height as the input.
        - `do_argmax`: Whether to perform an argmax operation on the predicted logits.
        """
        # TODO: the output for the default task depends on whether this is an image model or not

        if self.task == "defaultX":
            return OrderedDict(
                [
                    (
                        "last_hidden_state",
                        {
                            "description": "Sequence of hidden-states at the output of the last layer of the model",
                        }
                    ),
                    (
                        "pooler_output",
                        {
                            "description": "Last layer hidden-state of the first token of the sequence",
                        }
                    ),
                ]
            )

        if self.task == "default":
            return OrderedDict(
                [
                    (
                        "last_hidden_state",
                        {
                            "description": "Sequence of hidden-states at the output of the last layer of the model",
                        }
                    ),
                    (
                        "pooler_output",
                        {
                            "description": "Last layer hidden-state after a pooling operation on the spatial dimensions",
                        }
                    ),
                ]
            )

        if self.task in [
            "image-classification",
            "multiple-choice",
            "next-sentence-prediction",
            "sequence-classification",
        ]:
            return OrderedDict(
                [
                    (
                        "probabilities",
                        {
                            "description": "Probability of each category",
                        }
                    ),
                    (
                        "classLabel",
                        {
                            "description": "Category with the highest score",
                        }
                    ),
                ]
            )

        if self.task == "masked-im":
            return OrderedDict(
                [
                    (
                        "logits",
                        {
                            "description": "Prediction scores (before softmax)",
                        }
                    ),
                ]
            )

        if self.task in ["masked-lm", "token-classification"]:
            return OrderedDict(
                [
                    (
                        "token_scores",
                        {
                            "description": "Prediction scores for each vocabulary token (after softmax)",
                        }
                    ),
                ]
            )

        if self.task == "object-detection":
            return OrderedDict(
                [
                    (
                        "logits",
                        {
                            "description": "Classification logits (including no-object) for all queries",
                        }
                    ),
                    (
                        "pred_boxes",
                        {
                            "description": "Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height)",
                        }
                    ),
                ]
            )

        if self.task == "question-answering":
            return OrderedDict(
                [
                    (
                        "start_scores",
                        {
                            "description": "Span-start scores (after softmax)",
                        }
                    ),
                    (
                        "end_scores",
                        {
                            "description": "Span-end scores (after softmax)",
                        }
                    ),
                ]
            )

        if self.task == "semantic-segmentation":
            return OrderedDict(
                [
                    (
                        "classLabels",
                        {
                            "description": "Segmentation map",
                            "do_argmax": True,
                            "do_upsample": True,
                        }
                    ),
                ]
            )

        raise AssertionError("Unsupported task '{self.task}'")

        #TODO: maybe do it like ONNX where we just copy the dictionary (don't need the assert then):
        # common_outputs = self._tasks_to_common_outputs[self.task]
        # return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting

        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        return None

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

        input_defs = self.inputs
        dummy_inputs = {}

        if isinstance(preprocessor, PreTrainedTokenizerBase):
            input_name, input_config = input_defs.popitem(last=False)
            sequence_length = input_config.get("sequence_length", 128)

            if self.task == "multiple-choice":
                shape = (1, self._config.num_labels, sequence_length)
            else:
                shape = (1, sequence_length)

            input_ids = np.random.randint(0, preprocessor.vocab_size, shape)
            dummy_inputs[input_name] = input_ids

            # attention_mask
            if len(input_defs) > 0:
                input_name, input_config = input_defs.popitem(last=False)
                attention_mask = np.ones(shape, dtype=np.int64)
                dummy_inputs[input_name] = attention_mask

            # token_type_ids
            if len(input_defs) > 0:
                input_name, input_config = input_defs.popitem(last=False)
                token_type_ids = np.zeros(shape, dtype=np.int64)
                dummy_inputs[input_name] = token_type_ids

        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            if hasattr(preprocessor, "crop_size"):
                image_size = preprocessor.crop_size
            else:
                image_size = preprocessor.size

            if isinstance(image_size, tuple):
                image_width, image_height = image_size
            else:
                image_width = image_height = image_size

            pixel_values = np.random.rand(1, 3, image_height, image_width).astype(np.float32) * 2.0 - 1.0

            input_name, input_config = input_defs.popitem(last=False)
            dummy_inputs[input_name] = pixel_values

            # bool_masked_pos
            if self.task == "masked-im":
                num_patches = (self._config.image_size // self._config.patch_size) ** 2
                bool_masked_pos = np.random.randint(low=0, high=2, size=(1, num_patches)).astype(bool)
                input_name, input_config = input_defs.popitem(last=False)
                dummy_inputs[input_name] = bool_masked_pos

        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

        return dummy_inputs

    def patch_pytorch_ops(self) -> Mapping[str, Callable]:
        """
        Override this to provide implementation for PyTorch ops that the Core ML
        converter does not support.

        Returns:
            `Mapping[str, Callable]` of op names to PyTorch conversion functions.
        """
        return {}
