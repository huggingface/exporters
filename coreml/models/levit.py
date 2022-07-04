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
"""Core ML conversion for LeViT."""

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

from transformers import (
    LevitFeatureExtractor,
    LevitModel,
    LevitForImageClassification,
    LevitForImageClassificationWithTeacher,
)
from ..coreml_utils import *


if "reshape_as" in _TORCH_OPS_REGISTRY:
    del _TORCH_OPS_REGISTRY["reshape_as"]

@register_torch_op
def reshape_as(context, node):
    a = context[node.inputs[0]]
    b = context[node.inputs[1]]
    y = mb.shape(x=b)
    x = mb.reshape(x=a, shape=y, name=node.name)
    context.add(x)


class Wrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor):
        super().__init__()
        self.model = model.eval()
        self.feature_extractor = feature_extractor

    def forward(self, inputs):
        # Core ML's image preprocessing does not allow a different
        # scaling factor for each color channel, so do this manually.
        image_std = torch.tensor(self.feature_extractor.image_std).reshape(1, -1, 1, 1)
        inputs = inputs / image_std

        outputs = self.model(inputs, return_dict=False)

        if isinstance(self.model, LevitForImageClassification):
            return nn.functional.softmax(outputs[0], dim=1)

        if isinstance(self.model, LevitForImageClassificationWithTeacher):
            return outputs[0], outputs[1], outputs[2]  # logits, cls_logits, distillation_logits

        if isinstance(self.model, LevitModel):
            return outputs[0], outputs[1]  # last_hidden_state, pooler_output

        return None


def export(
    torch_model,
    feature_extractor: LevitFeatureExtractor,
    quantize: str = "float32",
    legacy: bool = False,
    **kwargs,
) -> ct.models.MLModel:
    if not isinstance(feature_extractor, LevitFeatureExtractor):
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    scale = 1.0 / 255
    bias = [
        -feature_extractor.image_mean[0],
        -feature_extractor.image_mean[1],
        -feature_extractor.image_mean[2],
    ]

    image_size = feature_extractor.size
    image_shape = (1, 3, image_size, image_size)
    pixel_values = torch.rand(image_shape) * 2.0 - 1.0
    example_input = [ pixel_values ]

    wrapper = Wrapper(torch_model, feature_extractor).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    # Run the PyTorch model, to get the shapes of the output tensors.
    with torch.no_grad():
        example_output = traced_model(*example_input)

    convert_kwargs = { }
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    if isinstance(torch_model, LevitForImageClassification):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    # pass any additional arguments to ct.convert()
    for key, value in kwargs.items():
        convert_kwargs[key] = value

    input_tensors = [ ct.ImageType(name="image", shape=image_shape, scale=scale, bias=bias,
                                   color_layout="RGB", channel_first=True) ]

    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    if isinstance(torch_model, LevitForImageClassification):
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(torch_model, LevitForImageClassificationWithTeacher):
        # Rename the outputs.
        output_names = get_output_names(spec)
        ct.utils.rename_feature(spec, output_names[0], "logits")
        ct.utils.rename_feature(spec, output_names[1], "cls_logits")
        ct.utils.rename_feature(spec, output_names[2], "distill_logits")

        set_multiarray_shape(get_output_named(spec, "logits"), example_output[0].shape)
        set_multiarray_shape(get_output_named(spec, "cls_logits"), example_output[1].shape)
        set_multiarray_shape(get_output_named(spec, "distill_logits"), example_output[2].shape)

        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["logits"] = "Prediction scores as the average of the `cls_logits` and `distill_logits`"
        mlmodel.output_description["cls_logits"] = "Prediction scores of the classification head"
        mlmodel.output_description["distill_logits"] = "Prediction scores of the distillation head"

    if isinstance(torch_model, LevitModel):
        # Rename the outputs.
        output_names = get_output_names(spec)
        ct.utils.rename_feature(spec, output_names[0], "last_hidden_state")
        ct.utils.rename_feature(spec, output_names[1], "pooler_output")

        set_multiarray_shape(get_output_named(spec, "last_hidden_state"), example_output[0].shape)
        set_multiarray_shape(get_output_named(spec, "pooler_output"), example_output[1].shape)

        mlmodel.input_description["image"] = "Image input"
        mlmodel.output_description["last_hidden_state"] = "Sequence of hidden-states at the output of the last layer of the model"
        mlmodel.output_description["pooler_output"] = "Last layer hidden-state after a pooling operation on the spatial dimensions"

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel
