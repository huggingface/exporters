<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Models that are / aren't supported by 🤗 Exporters

Only models that have a `ModelNameCoreMLConfig` object are currently supported.

If a model is not supported, this is either because there is some problem with the actual conversion process, or because we simply did not get around to writing a `CoreMLConfig` object for it.

## Supported models

Legend:

- ✅ = fully supported
- 😓 = works but with hacks
- ⚠️ = partially supported (for example no "with past" version)
- ❌ = errors during conversion
- ➖ = not supported
- ? = unknown

### Text Models

**BERT**

- ✅ BertModel
- ➖ BertForPreTraining
- ✅ BertForMaskedLM
- ✅ BertForMultipleChoice
- ✅ BertForNextSentencePrediction
- ✅ BertForQuestionAnswering
- ✅ BertForSequenceClassification
- ✅ BertForTokenClassification
- 😓 BertLMHeadModel: Appears to work OK when using `use_legacy_format`. However, when using MIL the conversion succeeds but the model may not actually run. It works OK with `use_past=False` or flexible sequence lengths, but blows up with `use_past=True` (Core ML takes forever to load the model, allocates 100+ GB of RAM and eventually crashes).

**BigBird**

- ? BigBirdModel
- ➖ BigBirdForPreTraining
- 😓 BigBirdForCausalLM: Appears to work OK when using `use_legacy_format`. However, when using MIL the conversion succeeds but the model may not actually run. It works OK with `use_past=False` or flexible sequence lengths, but blows up with `use_past=True` (Core ML takes forever to load the model, allocates 100+ GB of RAM and eventually crashes).
- ? BigBirdForMaskedLM
- ? BigBirdForMultipleChoice
- ? BigBirdForQuestionAnswering
- ? BigBirdForSequenceClassification
- ? BigBirdForTokenClassification

**CTRL**

- ✅ CTRLModel
- ✅ CTRLLMHeadModel
- ✅ CTRLForSequenceClassification

**DistilBERT**

- ✅ DistilBertModel
- ✅ DistilBertForMaskedLM
- ✅ DistilBertForMultipleChoice
- ✅ DistilBertForQuestionAnswering
- ✅ DistilBertForSequenceClassification
- ✅ DistilBertForTokenClassification


**ERNIE**

- ? ErnieModel
- ➖ ErnieForPreTraining
- 😓 ErnieForCausalLM: Appears to work OK when using `use_legacy_format`. However, when using MIL the conversion succeeds but the model may not actually run. It works OK with `use_past=False` or flexible sequence lengths, but blows up with `use_past=True` (Core ML takes forever to load the model, allocates 100+ GB of RAM and eventually crashes).
- ? ErnieForMaskedLM
- ? ErnieForMultipleChoice
- ? ErnieForNextSentencePrediction
- ? ErnieForQuestionAnswering
- ? ErnieForSequenceClassification
- ? ErnieForTokenClassification

**GPT2 / DistilGPT2**

Needs to be exported with `use_legacy_format=True`. Does not work with flexible sequence length and therefore does not support `use_past`.

- ✅ GPT2Model
- ➖ GPT2DoubleHeadsModel
- ✅ GPT2ForSequenceClassification
- ✅ GPT2ForTokenClassification
- ⚠️ GPT2LMHeadModel (no `use_past`)

**MobileBERT**

- ✅ MobileBertModel
- ➖ MobileBertForPreTraining
- ✅ MobileBertForMaskedLM
- ✅ MobileBertForMultipleChoice
- ✅ MobileBertForNextSentencePrediction
- ✅ MobileBertForQuestionAnswering
- ✅ MobileBertForSequenceClassification
- ✅ MobileBertForTokenClassification

**SqueezeBERT**

- ✅ SqueezeBertModel
- ✅ SqueezeBertForMaskedLM
- ✅ SqueezeBertForMultipleChoice
- ✅ SqueezeBertForQuestionAnswering
- ✅ SqueezeBertForSequenceClassification
- ✅ SqueezeBertForTokenClassification

### Vision Models

**BEiT**

- ✅ BeitModel
- ✅ BeitForImageClassification
- ✅ BeitForSemanticSegmentation
- ✅ BeitForMaskedImageModeling. Note: this model does not work with AutoModelForMaskedImageModeling and therefore the conversion script cannot load it, but converting from Python is supported.

**ConvNeXT**

- ✅ ConvNextModel
- ✅ ConvNextForImageClassification

**CvT**

- ✅ CvtModel
- ✅ CvtForImageClassification

**LeViT**

- ✅ LevitModel
- ✅ LevitForImageClassification
- ➖ LevitForImageClassificationWithTeacher

**MobileViT**

- ✅ MobileViTModel
- ✅ MobileViTForImageClassification
- ✅ MobileViTForSemanticSegmentation

**SegFormer**

- ✅ SegformerModel
- ✅ SegformerForImageClassification
- ✅ SegformerForSemanticSegmentation

**Vision Transformer (ViT)**

- ✅ ViTModel
- ✅ ViTForMaskedImageModeling
- ✅ ViTForImageClassification

**YOLOS**

- ✅ YolosModel
- ✅ YolosForObjectDetection

### Audio Models

None

### Multimodal Models

None

## Models that currently don't work

The following models are known to give errors when attempting conversion to Core ML format, or simply have not been tried yet.

### Text Models

ALBERT

**BART**

- ? BartModel
- ❌ BartForCausalLM: Conversion succeeds, but "Error computing NN outputs".
- ❌ BartForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.
- ? BartForQuestionAnswering
- ? BartForSequenceClassification

BARThez

BARTpho

BertGeneration

BertJapanese

Bertweet

**BigBirdPegasus**

- ❌ BigBirdPegasusForCausalLM: Core ML error "Error computing NN outputs"
- ❌ BigBirdPegasusForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**Blenderbot**

- ❌ BlenderbotForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**Blenderbot Small**

- ❌ BlenderbotSmallForConditionalGeneration:  Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**BLOOM** [TODO verify] Conversion error on a slicing operation.

BORT

ByT5

CamemBERT

CANINE

**CodeGen** [TODO verify] Conversion error on einsum.

ConvBERT

CPM

DeBERTa

DeBERTa-v2

DialoGPT

DPR

**ELECTRA**

- ❌ ElectraForCausalLM: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op. Also, `coreml_config.values_override` doesn't work to set `use_cache` to True for this model.

Encoder Decoder Models

ESM

FlauBERT

FNet

**FSMT**

- ❌ FSMTForConditionalGeneration. Encoder converts OK. For decoder, `Wrapper` outputs wrong size logits tensor; goes wrong somewhere in hidden states output from decoder when `return_dict=False`?

Funnel Transformer

GPT

**GPT Neo**. [TODO verify] Gives no errors during conversion but predicts wrong results, or NaN when `use_legacy_format=True`.

- GPTNeoModel
- GPTNeoForCausalLM
- GPTNeoForSequenceClassification

GPT NeoX

GPT NeoX Japanese

GPT-J

HerBERT

I-BERT

LayoutLM

**LED**

- ❌ LEDForConditionalGeneration: JIT trace fails with the error:

```python
RuntimeError: 0INTERNAL ASSERT FAILED at "/Users/distiller/project/pytorch/torch/csrc/jit/ir/alias_analysis.cpp":607, please report a bug to PyTorch. We don't have an op for aten::constant_pad_nd but it isn't a special case.  Argument types: Tensor, int[], bool,
```

LiLT

Longformer

**LongT5**

- ❌ LongT5ForConditionalGeneration: Conversion error:

```python
ValueError: In op, of type not_equal, named 133, the named input `y` must have the same data type as the named input `x`. However, y has dtype fp32 whereas x has dtype int32.
```

LUKE

**M2M100**

- ❌ M2M100ForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**MarianMT**

- ❌ MarianMTModel: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

MarkupLM

MBart and MBart-50

MegatronBERT

MegatronGPT2

mLUKE

MPNet

**MT5**

- ❌ MT5ForConditionalGeneration: Converter error "User defined pattern has more than one final operation"

**MVP**

- ❌ MvpForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**NEZHA** [TODO verify] Conversion error on a slicing operation.

NLLB

Nyströmformer

**OPT** [TODO verify] Conversion error on a slicing operation.

**Pegasus**

- ❌ PegasusForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**PEGASUS-X**

- ❌ PegasusXForConditionalGeneration: "AttributeError: 'list' object has no attribute 'val'" in `pad` op. Maybe: needs `remainder` op (added recently in coremltools dev version).

PhoBERT

**PLBart**

- ❌ PLBartForConditionalGeneration: Encoder converts OK (tested only `use_past=False`). Decoder gives "Error computing NN outputs" with flexible sequence length.

**ProphetNet**

- ❌ ProphetNetForConditionalGeneration. Conversion error:

```python
ValueError: Op "input.3" (op_type: clip) Input x="position_ids" expects tensor or scalar of dtype from type domain ['fp16', 'fp32'] but got tensor[1,is4273,int32]
```

QDQBert

RAG

REALM

**Reformer**

- ❌ ReformerModelWithLMHead: does not have `past_key_values` but `past_buckets_states`

**RemBERT**

- ❌ RemBertForCausalLM: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op. Also error adding a constant value to the MIL graph?

RetriBERT

**RoBERTa**

- ❌ RobertaForCausalLM: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op.

**RoFormer**

- ❌ RoFormerForCausalLM: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op. Also error adding a constant value to the MIL graph?

**Splinter**

- ❌ SplinterModel: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op. Also error adding a constant value to the MIL graph?

**T5**

- ❌ T5ForConditionalGeneration: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op.

T5v1.1

TAPAS

TAPEX

Transformer XL

UL2

**XGLM** [TODO verify] Conversion error on a slicing operation.

XLM

**XLM-ProphetNet**

- XLMProphetNetForConditionalGeneration: Conversion error:

```python
ValueError: Op "input.3" (op_type: clip) Input x="position_ids" expects tensor or scalar of dtype from type domain ['fp16', 'fp32'] but got tensor[1,is4506,int32]
```

XLM-RoBERTa

XLM-RoBERTa-XL

**XLNet** [TODO verify] Conversion error.

YOSO

### Vision Models

Conditional DETR

Deformable DETR

DeiT

**DETR** [TODO verify] The conversion completes without errors but the Core ML compiler cannot load the model. "Invalid operation output name: got 'tensor' when expecting token of type 'ID'"

DiT

DPT

GLPN

ImageGPT

MaskFormer

PoolFormer

RegNet

ResNet

**Swin Transformer** [TODO verify] The PyTorch graph contains unsupported operations: remainder, roll, adaptive_avg_pool1d. (Some of these may be supported in latest dev version.)

Swin Transformer V2

VAN

VideoMAE

ViTMAE

ViTMSN

### Audio Models

**Hubert** [TODO verify] Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.

MCTCT

**SEW** [TODO verify] Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.

SEW-D

**Speech2Text** [TODO verify] The "glu" op is not supported by coremltools. Should be possible to solve by defining a `@register_torch_op` function. (Update: should be supported in dev version now.)

Speech2Text2

**UniSpeech** [TODO verify] Missing op for `_weight_norm` (possible to work around), also same Core ML compiler error as DETR.

UniSpeech-SAT

**Wav2Vec2** [TODO verify] Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.

Wav2Vec2-Conformer

Wav2Vec2Phoneme

**WavLM** [TODO verify] Missing ops for `_weight_norm`, `add_`, `full_like`.

Whisper

XLS-R

XLSR-Wav2Vec2

### Multimodal Models

CLIP

**Data2Vec**

- ❌ Data2VecTextForCausalLM: "AttributeError: 'list' object has no attribute 'val'" in `repeat` op

- Data2VecAudio: [TODO verify] The conversion completes without errors but the Core ML compiler cannot load the model.

Donut

FLAVA

**GroupViT** [TODO verify] Conversion issue with `scatter_along_axis` operation.

LayoutLMV2

LayoutLMV3

LayoutXLM

LXMERT

OWL-ViT

Perceiver

Speech Encoder Decoder Models

TrOCR

ViLT

Vision Encoder Decoder Models

Vision Text Dual Encoder

VisualBERT

X-CLIP
