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


class FalconCoreMLConfig(CoreMLConfig):
    modality = "text"

    def patch_pytorch_ops(self):
        # Copied from https://github.com/apple/coremltools/blob/b2f719075dc5bc19280a3045c1762d7d32bd3fdc/coremltools/converters/mil/frontend/torch/ops.py#L4326
        # with fallback of `bfloat16` to `float32`.
        def to(context, node):
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil.mil import types
            from coremltools.converters.mil.frontend.torch.ops import (
                _get_inputs,
                NUMPY_DTYPE_TO_TORCH_NUM,
                NUM_TO_TORCH_DTYPE,
                NUM_TO_DTYPE_STRING,
                NUM_TO_NUMPY_DTYPE,
                TORCH_DTYPE_TO_NUM,
            )
            from coremltools.converters.mil.mil.types import nptype_from_builtin
            from coremltools.converters.mil.mil.var import Var
            import numpy as _np
            import torch

            inputs = _get_inputs(context, node)

            # There are a lot of variants of `to` op.
            # - When len(inputs) is 7 or 8, we only care about the first two params (input and dtype).
            # - When len(inputs) == 6, the parameter is (input, _, dtype, non_blocking, copy, memory_format)
            # - When len(inputs) == 5, the parameter is (input, dtype, non_blocking, copy, memory_format)
            # - When len(inputs) == 4, the parameter is (input, dtype, non_blocking, copy)
            # - When len(inputs) == 3, the parameter is (input, non_blocking, copy)
            # We only use `input` and `dtype`, and `non_blocking` and `copy` are unused.
            _input = inputs[0]

            inputs_len = len(inputs)
            if inputs_len in (4, 5, 7, 8):
                target_dtype = inputs[1]
            elif inputs_len == 6:
                target_dtype = inputs[2]
            elif inputs_len <= 3:
                target_dtype = None
            else:
                raise ValueError(
                    "Received invalid arguments for PyTorch conversion of op {}".format(node)
                )

            if target_dtype is None:
                # When target_dtype is None, it means the input's dtype is already the target dtype.
                context.add(_input, torch_name=node.name)
                return
            elif types.is_scalar(target_dtype.sym_type) and target_dtype.val is not None:
                dtype = target_dtype.val
            else:
                # When the val of dtype is not available, bridge from the np dtype.
                np_type = nptype_from_builtin(target_dtype.dtype)
                dtype = NUMPY_DTYPE_TO_TORCH_NUM[np_type]

            if dtype in NUM_TO_TORCH_DTYPE:
                torch_dtype = NUM_TO_TORCH_DTYPE[dtype]
            else:
                # Fallback `bfloat32` to `fp32` for now.
                torch_dtype = torch.float32

            if isinstance(_input, Var) and _input.can_be_folded_to_const():
                # numpy -> torch -> torch cast -> numpy
                # This path is needed to use the mapping of passed in dtypes to torch dtypes.
                casted_input = torch.tensor(_input.val).type(torch_dtype).cpu().numpy()
                res = mb.const(val=casted_input, name=node.name)
            else:
                if dtype in NUM_TO_DTYPE_STRING:
                    res = mb.cast(x=_input, dtype=NUM_TO_DTYPE_STRING[dtype], name=node.name)
                else:
                    # For dtype that is not supported by mb.cast, we do it in best-effort to cast it to int
                    # or float based on the dtype.
                    np_dtype = NUM_TO_NUMPY_DTYPE[dtype]
                    if _np.issubdtype(np_dtype, _np.integer):
                        res = mb.cast(x=_input, dtype="int32", name=node.name)
                    elif _np.issubdtype(np_dtype, _np.floating):
                        res = mb.cast(x=_input, dtype="fp32", name=node.name)
                    else:
                        raise ValueError(f"Unsupported op {node} with target dtype {np_dtype}")
            context.add(res)

        # Workaround until https://github.com/apple/coremltools/pull/2046 is released 
        def numpy_t(context, node):
            from coremltools.converters.mil import Builder as mb

            assert len(node.outputs) == 1
            assert len(node.inputs) == 1

            x = context[node.inputs[0]]
            assert len(x.shape) == 2

            res = mb.transpose(x=x, perm=[1, 0], name=node.name)
            context.add(res)

        return {"to": to, "numpy_t": numpy_t}

    @property
    def atol_for_validation(self) -> float:
        # Possibly required because of internal `bfloat16` conversions in the model
        # float32 conversion requires ~0.03, whereas `float16` requires ~0.1
        return 0.1


class GPT2CoreMLConfig(CoreMLConfig):
    modality = "text"


class GPTBigcodeCoreMLConfig(CoreMLConfig):
    modality = "text"


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


class MistralCoreMLConfig(CoreMLConfig):
    modality = "text"

    def patch_pytorch_ops(self):
        # Workaround for https://github.com/apple/coremltools/pull/2017
        def log(context, node):
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil.mil import types

            a = context[node.inputs[0]]
            if types.is_int(a.dtype):
                a = mb.cast(x=a, dtype="fp32")
            x = mb.log(x=a, name=node.name)
            context.add(x)

        return {"log": log}


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
