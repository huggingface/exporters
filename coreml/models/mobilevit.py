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

import numpy as np
import coremltools as ct

import torch
import torch.nn
import torch.nn.functional as F

from transformers import MobileViTFeatureExtractor, MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation
from ..coreml_utils import *


# Note: the MobileViTForImageClassification model has the usual `classLabel` 
# and `classLabel_probs` outputs, but also a hidden `var_1385` output with the 
# softmax results. Not sure why, but it doesn't hurt anything to keep it.


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, inputs):
        outputs = self.model(inputs)

        if isinstance(self.model, MobileViTForImageClassification):
            return F.softmax(outputs.logits, dim=1)

        if isinstance(self.model, MobileViTForSemanticSegmentation):
            return outputs.logits

        if isinstance(self.model, MobileViTModel):
            return outputs.last_hidden_state, outputs.pooler_output

        return None
    

def export(torch_model, preprocessor: MobileViTFeatureExtractor, quantize: str = "float32") -> ct.models.MLModel:
    if not isinstance(preprocessor, MobileViTFeatureExtractor):
        raise ValueError(f"Unknown preprocessor: {preprocessor}")

    wrapper = Wrapper(torch_model)

    scale = 1.0 / 255
    bias = [ 0, 0, 0 ]

    img_size = preprocessor.crop_size
    img_shape = (1, 3, img_size, img_size)
    example_input = torch.rand(img_shape) * 2.0 - 1.0

    traced_model = torch.jit.trace(wrapper, example_input, strict=False)

    convert_kwargs = { }
    if isinstance(torch_model, MobileViTForImageClassification):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=img_shape, scale=scale, bias=bias, 
                             color_layout="BGR", channel_first=True)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32,
        **convert_kwargs,
    )

    spec = mlmodel._spec

    if isinstance(torch_model, MobileViTForImageClassification):
        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["classLabel_probs"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"


    # TODO: output name + shape for segmentation model
    # TODO: add class labels to metadata (since this is not a classifier)


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

    user_defined_metadata = {
        "transformers_version": torch_model.config.transformers_version,
    }
    spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    return mlmodel
