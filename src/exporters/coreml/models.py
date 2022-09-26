# coding=utf-8
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

from collections import OrderedDict

from .config import (
    CoreMLConfig,
    InputDescription,
    OutputDescription,
)


class BeitCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs

    @property
    def atol_for_validation(self) -> float:
        return 0.01


class BertCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        # TODO: BERT gives conversion error with flexible input shape.
        input_descs["input_ids"].sequence_length = 128
        return input_descs

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class ConvNextCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs

    @property
    def atol_for_validation(self) -> float:
        return 1e-3


class CTRLCoreMLConfig(CoreMLConfig):
    modality = "text"


class CvtCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        if self.task == "default":
            return OrderedDict(
                [
                    (
                        "last_hidden_state",
                        OutputDescription(
                            "last_hidden_state",
                            "Sequence of hidden-states at the output of the last layer of the model",
                        )
                    ),
                    (
                        "cls_token_value",
                        OutputDescription(
                            "cls_token_value",
                            "Classification token at the output of the last layer of the model",
                        )
                    ),
                ]
            )
        else:
            return super().outputs

    def patch_pytorch_ops(self):
        # coremltools does support einsum but not the equation "bhlt,bhtv->bhlv"
        # so override the implementation of this operation
        def einsum(context, node):
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil.frontend._utils import build_einsum_mil

            a = context[node.inputs[1]][0]
            b = context[node.inputs[1]][1]
            equation = context[node.inputs[0]].val

            if equation == "bhlt,bhtv->bhlv":
                x = mb.matmul(x=a, y=b, transpose_x=False, transpose_y=False, name=node.name)
            else:
                x = build_einsum_mil(a, b, equation, node.name)

            context.add(x)

        return {"einsum": einsum}

    @property
    def atol_for_validation(self) -> float:
        return 0.01


class DistilBertCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        if self.task == "multiple-choice":
            return OrderedDict(
                [
                    (
                        "input_ids",
                        InputDescription(
                            "input_ids",
                            "Indices of input sequence tokens in the vocabulary",
                            sequence_length=(1, 128),
                        )
                    ),
                    (
                        "attention_mask",
                        InputDescription(
                            "attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                ]
            )
        else:
            return super().inputs


class GPT2CoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        # TODO: coremltools blows up and uses infinite RAM with flexible input shape
        input_descs["input_ids"].sequence_length = 128
        return input_descs

    @property
    def use_legacy_format(self) -> bool:
        return True


class GPTNeoCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        # TODO: coremltools blows up and uses infinite RAM with flexible input shape
        input_descs["input_ids"].sequence_length = 128
        return input_descs


class LevitCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs

    def patch_pytorch_ops(self):
        def reshape_as(context, node):
            from coremltools.converters.mil import Builder as mb

            a = context[node.inputs[0]]
            b = context[node.inputs[1]]
            y = mb.shape(x=b)
            x = mb.reshape(x=a, shape=y, name=node.name)
            context.add(x)

        return {"reshape_as": reshape_as}

    @property
    def atol_for_validation(self) -> float:
        return 0.01


class MobileBertCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class MobileViTCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        input_descs["pixel_values"].color_layout = "BGR"
        return input_descs

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class SegformerCoreMLConfig(CoreMLConfig):
    modality = "vision"


class SqueezeBertCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class ViTCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class YolosCoreMLConfig(CoreMLConfig):
    modality = "vision"

    def patch_pytorch_ops(self):
        # There is no bicubic upsampling in Core ML, so we'll have to use bilinear.
        # Still seems to work well enough. Note: the bilinear resize is applied to
        # constant tensors, so we could actually remove this op completely!
        def upsample_bicubic2d(context, node):
            from coremltools.converters.mil import Builder as mb

            a = context[node.inputs[0]]
            b = context[node.inputs[1]]
            x = mb.resize_bilinear(x=a, target_size_height=b.val[0], target_size_width=b.val[1], name=node.name)
            context.add(x)

        return {"upsample_bicubic2d": upsample_bicubic2d}

    @property
    def atol_for_validation(self) -> float:
        # because of bilinear instead of bicubic, atol must be very large here
        return 10
