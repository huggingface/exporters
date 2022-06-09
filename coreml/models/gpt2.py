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
"""Core ML conversion for GPT2."""

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from transformers import PreTrainedTokenizerBase, GPT2LMHeadModel
from ..coreml_utils import *


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, inputs):
        outputs = self.model(
            inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        if is_any_instance(self.model, [GPT2LMHeadModel]):
            # Performing a softmax here will give nan for the NeuralNetwork version,
            # so output logits instead. Note: The ML Program version does the softmax
            # without problems, but produces completely incorrect logits.
            #return nn.functional.softmax(outputs[0], dim=2)
            return outputs[0]

        return None


def export(
    torch_model,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int = 64,
    quantize: str = "float32",
    legacy: bool = False,
    **kwargs,
) -> ct.models.MLModel:
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    example_input = torch.randint(tokenizer.vocab_size, (1, sequence_length))

    wrapper = Wrapper(torch_model).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    convert_kwargs = {}
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    # pass any additional arguments to ct.convert()
    for key, value in kwargs.items():
        convert_kwargs[key] = value

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=example_input.shape, dtype=np.int32)],
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    if is_any_instance(torch_model, [GPT2LMHeadModel]):
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "output_logits")
        set_multiarray_shape(output, (1, sequence_length, tokenizer.vocab_size))

        mlmodel.input_description["input_ids"] = "Indices of input sequence tokens in the vocabulary"
        mlmodel.output_description["output_logits"] = "Prediction scores of the language modeling head (scores for each vocabulary token before softmax)"

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel
