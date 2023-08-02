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


def patch_common_pytorch_ops():
    """
    Workarounds for issues that haven't been fixed yet in coremltools that
    affect many of our models.
    """
    # from coremltools.converters.mil import Builder as mb
    return {}


class BartCoreMLConfig(CoreMLConfig):
    modality = "text"


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
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class BigBirdCoreMLConfig(CoreMLConfig):
    modality = "text"


class BigBirdPegasusCoreMLConfig(CoreMLConfig):
    modality = "text"


class BlenderbotCoreMLConfig(CoreMLConfig):
    modality = "text"


class BlenderbotSmallCoreMLConfig(CoreMLConfig):
    modality = "text"


class BloomCoreMLConfig(CoreMLConfig):
    modality = "text"


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

    def patch_pytorch_ops(self):
        """Implement lift_fresh as a noop, unless it's already available in a future update."""
        import coremltools.converters.mil.frontend.torch.ops as ops
        if hasattr(ops, "lift_fresh"):
            return {}

        def lift_fresh(context, node):
            a = context[node.inputs[0]]
            context.add(a, node.name)

        return {"lift_fresh": lift_fresh}


class CvtCoreMLConfig(CoreMLConfig):
    modality = "vision"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        if self.task == "feature-extraction":
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


class Data2VecTextCoreMLConfig(CoreMLConfig):
    modality = "text"


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
                            sequence_length=self.input_ids_sequence_length,
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


class ErnieCoreMLConfig(CoreMLConfig):
    modality = "text"


class GPT2CoreMLConfig(CoreMLConfig):
    modality = "text"


class GPTBigcodeCoreMLConfig(CoreMLConfig):
    modality = "text"

    def patch_pytorch_ops(self):
        # Until https://github.com/apple/coremltools/pull/1911 is released
        def gelu(context, node):
            from coremltools.converters.mil import Builder as mb

            inputs = [context[name] for name in node.inputs]
            assert len(inputs) in (1, 2)
            if len(inputs) == 2:
                approximate = inputs[1].val
                if approximate == "tanh":
                    approximate = "TANH_APPROXIMATION"
                elif approximate == "none":
                    approximate = "EXACT"
            else:
                approximate = None
            res = mb.gelu(x=inputs[0], mode=approximate, name=node.name)
            context.add(res)

        # Until https://github.com/apple/coremltools/pull/1910 is released
        def mul(context, node):
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil.mil import types
            from coremltools.converters.mil.mil.ops.defs._utils import promote_input_dtypes

            inputs = [context[name] for name in node.inputs]
            if len(inputs) != 2:
                raise ValueError(f"Expected two inputs for `mul`, got {len(inputs)} instead.")
            x, y = promote_input_dtypes(inputs)
            if types.is_bool(x.dtype) and types.is_bool(y.dtype):
                res = mb.logical_and(x=x, y=y, name=node.name)
            else:
                res = mb.mul(x=x, y=y, name=node.name)
            context.add(res)

        return {"gelu": gelu, "mul": mul}


class GPTJCoreMLConfig(CoreMLConfig):
    modality = "text"

    def patch_pytorch_ops(self):
        # https://github.com/apple/coremltools/issues/1852
        def einsum(context, node):
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil.frontend._utils import build_einsum_mil
            from coremltools.converters.mil.mil import types

            a = context[node.inputs[1]][0]
            b = context[node.inputs[1]][1]
            equation = context[node.inputs[0]].val
            equation = "".join(equation.split(" "))
            if equation == "i,j->ij" and types.is_int(a.dtype):
                a = mb.cast(x=a, dtype="fp32")
            x = build_einsum_mil(a, b, equation, node.name)

            context.add(x)

        return {"einsum": einsum}


class GPTNeoCoreMLConfig(CoreMLConfig):
    modality = "text"


class GPTNeoXCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        # Flexible shapes are incompatible with gather (https://github.com/huggingface/exporters/issues/43)
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


class LlamaCoreMLConfig(CoreMLConfig):
    modality = "text"


class M2M100CoreMLConfig(CoreMLConfig):
    modality = "text"


class MarianMTCoreMLConfig(CoreMLConfig):
    modality = "text"


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


class MvpCoreMLConfig(CoreMLConfig):
    modality = "text"


class PegasusCoreMLConfig(CoreMLConfig):
    modality = "text"


class PLBartCoreMLConfig(CoreMLConfig):
    modality = "text"


class RobertaCoreMLConfig(CoreMLConfig):
    modality = "text"


class RoFormerCoreMLConfig(CoreMLConfig):
    modality = "text"


class SegformerCoreMLConfig(CoreMLConfig):
    modality = "vision"


class SplinterCoreMLConfig(CoreMLConfig):
    modality = "text"


class SqueezeBertCoreMLConfig(CoreMLConfig):
    modality = "text"

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        output_descs = super().outputs
        self._add_pooler_output(output_descs)
        return output_descs


class T5CoreMLConfig(CoreMLConfig):
    modality = "text"
    
    @property
    def _input_descriptions(self) -> OrderedDict[str, InputDescription]:
        if self.task == "feature-extraction":
            return OrderedDict(
                [
                    (
                        "input_ids",
                        InputDescription(
                            "input_ids",
                            "Indices of input sequence tokens in the vocabulary",
                            sequence_length=self.input_ids_sequence_length,
                        )
                    ),
                    (
                        "attention_mask",
                        InputDescription(
                            "attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                    (
                        "decoder_input_ids",
                        InputDescription(
                            "decoder_input_ids",
                            "Indices of decoder input sequence tokens in the vocabulary",
                        )
                    ),
                    (
                        "decoder_attention_mask",
                        InputDescription(
                            "decoder_attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                ]
            )
        return super()._input_descriptions


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
