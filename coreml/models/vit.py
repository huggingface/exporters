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
"""Core ML conversion for Vision Transformer (ViT)."""

import numpy as np
import coremltools as ct

import torch
from torch import nn

from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification
from ..coreml_utils import *


# Note: the ViTForImageClassification model has the usual `classLabel` and
# `classLabel_probs` outputs, but also a hidden `var_992` output with the 
# softmax results. Not sure why, but it doesn't hurt anything to keep it.


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, inputs):
        outputs = self.model(inputs)

        if isinstance(self.model, ViTForImageClassification):
            return nn.functional.softmax(outputs.logits, dim=1)

        if isinstance(self.model, ViTModel):
            if self.model.pooler is not None:
                return outputs.last_hidden_state, outputs.pooler_output
            else:
                return outputs.last_hidden_state

        return None
    

def export(torch_model, preprocessor: ViTFeatureExtractor, quantize: str = "float32") -> ct.models.MLModel:
    if not isinstance(preprocessor, ViTFeatureExtractor):
        raise ValueError(f"Unknown preprocessor: {preprocessor}")

    wrapper = Wrapper(torch_model)

    scale = 1.0 / (preprocessor.image_std[0] * 255)
    bias = [
        -preprocessor.image_mean[0] / preprocessor.image_std[0],
        -preprocessor.image_mean[1] / preprocessor.image_std[1],
        -preprocessor.image_mean[2] / preprocessor.image_std[2],
    ]

    image_size = preprocessor.size
    image_shape = (1, 3, image_size, image_size)
    example_input = torch.rand(image_shape) * 2.0 - 1.0

    traced_model = torch.jit.trace(wrapper, example_input, strict=False)

    convert_kwargs = { }
    if isinstance(torch_model, ViTForImageClassification):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=image_shape, scale=scale, bias=bias, 
                             color_layout="RGB", channel_first=True)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32,
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    if isinstance(torch_model, ViTForImageClassification):
        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["classLabel_probs"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(torch_model, ViTModel):
        # Rename the output from the pooler.
        if torch_model.pooler is not None:
            output_names = get_output_names(spec)
            for output_name in output_names:
                if output_name != "hidden_states":
                    ct.utils.rename_feature(spec, output_name, "pooler_output")

        # Fill in the shapes for the output tensors.
        with torch.no_grad():
            temp = traced_model(example_input)

        if torch_model.pooler is not None:
            hidden_shape = temp[0].shape
            pooler_shape = temp[1].shape
            set_multiarray_shape(get_output_named(spec, "hidden_states"), hidden_shape)
            set_multiarray_shape(get_output_named(spec, "pooler_output"), pooler_shape)            
            mlmodel.output_description["pooler_output"] = "Output from the global pooling layer"
        else:
            hidden_shape = temp.shape
            set_multiarray_shape(get_output_named(spec, "hidden_states"), hidden_shape)

        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["hidden_states"] = "Hidden states from the last layer"

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    return mlmodel
