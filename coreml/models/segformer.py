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
"""Core ML conversion for SegFormer."""

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from transformers import SegformerFeatureExtractor, SegformerModel, SegformerForImageClassification, SegformerForSemanticSegmentation
from ..coreml_utils import *


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

        outputs = self.model(inputs)

        if isinstance(self.model, SegformerForImageClassification):
            return nn.functional.softmax(outputs.logits, dim=1)

        if isinstance(self.model, SegformerForSemanticSegmentation):
            x = nn.functional.interpolate(outputs.logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            x = x.argmax(1)
            return x

        if isinstance(self.model, SegformerModel):
            return outputs.last_hidden_state

        return None
    

def export(
    torch_model, 
    feature_extractor: SegformerFeatureExtractor, 
    quantize: str = "float32",
    legacy: bool = False,
) -> ct.models.MLModel:
    if not isinstance(feature_extractor, SegformerFeatureExtractor):
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    wrapper = Wrapper(torch_model, feature_extractor).eval()

    scale = 1.0 / 255
    bias = [
        -feature_extractor.image_mean[0],
        -feature_extractor.image_mean[1],
        -feature_extractor.image_mean[2],
    ]

    image_size = feature_extractor.size
    image_shape = (1, 3, image_size, image_size)
    example_input = torch.rand(image_shape) * 2.0 - 1.0

    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    convert_kwargs = { }
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    if isinstance(torch_model, SegformerForImageClassification):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=image_shape, scale=scale, bias=bias, 
                             color_layout="RGB", channel_first=True)],
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    if isinstance(torch_model, SegformerForImageClassification):
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(torch_model, SegformerForSemanticSegmentation):
        # Rename the segmentation output and fill in its shape.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "classLabels")
        set_multiarray_shape(output, (1, image_size, image_size))

        mlmodel.input_description["image"] = "Image input"
        mlmodel.output_description["classLabels"] = "Segmentation map"

        labels = get_labels_as_list(torch_model)
        user_defined_metadata["classes"] = ",".join(labels)

    if isinstance(torch_model, SegformerModel):
        # Rename the output and fill in its shape.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "last_hidden_state")

        with torch.no_grad():
            temp = traced_model(example_input)

        set_multiarray_shape(get_output_named(spec, "last_hidden_state"), temp.shape)

        mlmodel.input_description["image"] = "Image input"
        mlmodel.output_description["last_hidden_state"] = "Hidden states from the last layer"

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel
