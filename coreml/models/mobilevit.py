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
"""Core ML conversion for MobileViT."""

import json
import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from transformers import MobileViTFeatureExtractor, MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation
from ..coreml_utils import *


class Wrapper(torch.nn.Module):
    def __init__(self, model, do_upsample, do_argmax):
        super().__init__()
        self.model = model.eval()
        self.do_upsample = do_upsample
        self.do_argmax = do_argmax

    def forward(self, inputs):
        outputs = self.model(inputs, return_dict=False)

        if isinstance(self.model, MobileViTForImageClassification):
            return nn.functional.softmax(outputs[0], dim=1)

        if isinstance(self.model, MobileViTForSemanticSegmentation):
            x = outputs[0]
            if self.do_upsample:
                x = nn.functional.interpolate(x, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            if self.do_argmax:
                x = x.argmax(1)
            return x

        if isinstance(self.model, MobileViTModel):
            return outputs[0], outputs[1]  # last_hidden_state, pooler_output

        return None


def export(
    torch_model,
    feature_extractor: MobileViTFeatureExtractor,
    do_upsample: bool = True,
    do_argmax: bool = True,
    quantize: str = "float32",
    legacy: bool = False,
    **kwargs,
) -> ct.models.MLModel:
    if not isinstance(feature_extractor, MobileViTFeatureExtractor):
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    wrapper = Wrapper(torch_model, do_upsample, do_argmax).eval()

    scale = 1.0 / 255
    bias = [ 0.0, 0.0, 0.0 ]

    image_size = feature_extractor.crop_size
    image_shape = (1, 3, image_size, image_size)
    example_input = torch.rand(image_shape) * 2.0 - 1.0

    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    convert_kwargs = { }
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    if isinstance(torch_model, MobileViTForImageClassification):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    # pass any additional arguments to ct.convert()
    for key, value in kwargs.items():
        convert_kwargs[key] = value

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=image_shape, scale=scale, bias=bias,
                             color_layout="BGR", channel_first=True)],
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    if isinstance(torch_model, MobileViTForImageClassification):
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(torch_model, MobileViTForSemanticSegmentation):
        new_output_name = "classLabels" if do_argmax else "probabilities"

        # Rename the segmentation output.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, new_output_name)

        # Fill in the shapes for the output tensors.
        with torch.no_grad():
            temp = traced_model(example_input)

        set_multiarray_shape(output, temp.shape)

        mlmodel.input_description["image"] = "Image input"
        mlmodel.output_description[new_output_name] = "Segmentation map"

        labels = get_labels_as_list(torch_model)
        user_defined_metadata["classes"] = ",".join(labels)

        # Make the model available in Xcode's previewer.
        mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
        mlmodel.user_defined_metadata["com.apple.coreml.model.preview.params"] = json.dumps({"labels": labels})

    if isinstance(torch_model, MobileViTModel):
        # Rename the pooled output.
        output_names = get_output_names(spec)
        for output_name in output_names:
            if output_name != "last_hidden_state":
                ct.utils.rename_feature(spec, output_name, "pooler_output")

        mlmodel.input_description["image"] = "Image input"
        mlmodel.output_description["last_hidden_state"] = "Hidden states from the last layer"
        mlmodel.output_description["pooler_output"] = "Output from the global pooling layer"

        # Fill in the shapes for the output tensors.
        with torch.no_grad():
            temp = traced_model(example_input)

        hidden_shape = temp[0].shape
        pooler_shape = temp[1].shape
        set_multiarray_shape(get_output_named(spec, "last_hidden_state"), hidden_shape)
        set_multiarray_shape(get_output_named(spec, "pooler_output"), pooler_shape)

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel
