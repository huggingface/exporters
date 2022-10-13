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
- ⚠️ BertLMHeadModel
    - ❌ when `use_past=True`, "AttributeError: 'list' object has no attribute 'val'" on `repeat` op

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

BART

BARThez

BARTpho

BertGeneration

BertJapanese

Bertweet

**BigBird** "AttributeError: 'list' object has no attribute 'val'" in `reshape` op [is fixed now?]

**BigBirdPegasus** "AttributeError: 'list' object has no attribute 'val'" on the decoder [is fixed now?]

**Blenderbot** "AttributeError: 'list' object has no attribute 'val'" [is fixed now?]

**Blenderbot Small** "AttributeError: 'list' object has no attribute 'val'" [is fixed now?]

**BLOOM** Conversion error on a slicing operation.

BORT

ByT5

CamemBERT

CANINE

**CodeGen** Conversion error on einsum.

ConvBERT

CPM

DeBERTa

DeBERTa-v2

DialoGPT

DPR

**ELECTRA** "AttributeError: 'list' object has no attribute 'val'" in `reshape` op [is fixed now?]

Encoder Decoder Models

**ERNIE** Conversion error on a slicing operation. `is_decoder` model: "AttributeError: 'list' object has no attribute 'val'" in `reshape` op [is fixed now?]

ESM

FlauBERT

FNet

**FSMT** Wrapper outputs wrong size logits tensor; goes wrong somewhere in hidden states output from decoder when `return_dict=False`?

Funnel Transformer

GPT

**GPT Neo**. Gives no errors during conversion but predicts wrong results, or NaN when `use_legacy_format=True`.

GPT NeoX

GPT NeoX Japanese

GPT-J

HerBERT

I-BERT

LayoutLM

**LED** JIT trace fails

LiLT

Longformer

**LongT5** Conversion error on a `select` op?

LUKE

**M2M100** "AttributeError: 'list' object has no attribute 'val'" [is fixed now?] / "Invalid operation output name: got 'tensor' when expecting token of type 'ID'"

**MarianMT** "AttributeError: 'list' object has no attribute 'val'" [is fixed now?]

MarkupLM

MBart and MBart-50

MegatronBERT

MegatronGPT2

mLUKE

MPNet

MT5

**MVP** "Invalid operation output name: got 'tensor' when expecting token of type 'ID'"

**NEZHA** Conversion error on a slicing operation.

NLLB

Nyströmformer

**OPT** Conversion error on a slicing operation.

**Pegasus** "Invalid operation output name: got 'tensor' when expecting token of type 'ID'"

**PEGASUS-X** needs `remainder` op (added recently in dev version)

PhoBERT

**PLBart** "Invalid operation output name: got 'tensor' when expecting token of type 'ID'"

**ProphetNet** Conversion error on `clip` op.

QDQBert

RAG

REALM

Reformer

**RemBERT** Error adding a constant value to the MIL graph.

RetriBERT

**RoBERTa** "AttributeError: 'list' object has no attribute 'val'" in `reshape` op [is fixed now?]

**RoFormer** Error adding a constant value to the MIL graph.

**Splinter**  Error adding a constant value to the MIL graph.

**T5** The PyTorch graph contains unsupported operations.

T5v1.1

TAPAS

TAPEX

Transformer XL

UL2

**XGLM** Conversion error on a slicing operation.

XLM

**XLM-ProphetNet** Conversion error on `clip` op.

XLM-RoBERTa

XLM-RoBERTa-XL

**XLNet** Conversion error.

YOSO

### Vision Models

Conditional DETR

Deformable DETR

DeiT

**DETR** The conversion completes without errors but the Core ML compiler cannot load the model. "Invalid operation output name: got 'tensor' when expecting token of type 'ID'"

DiT

DPT

GLPN

ImageGPT

MaskFormer

PoolFormer

RegNet

ResNet

**Swin Transformer** The PyTorch graph contains unsupported operations: remainder, roll, adaptive_avg_pool1d. (Some of these may be supported in latest dev version.)

Swin Transformer V2

VAN

VideoMAE

ViTMAE

ViTMSN

### Audio Models

**Hubert** Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.

MCTCT

**SEW** Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.

SEW-D

**Speech2Text** The "glu" op is not supported by coremltools. Should be possible to solve by defining a `@register_torch_op` function. (Update: should be supported in dev version now.)

Speech2Text2

**UniSpeech** Missing op for `_weight_norm` (possible to work around), also same Core ML compiler error as DETR.

UniSpeech-SAT

**Wav2Vec2** Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.

Wav2Vec2-Conformer

Wav2Vec2Phoneme

**WavLM** Missing ops for `_weight_norm`, `add_`, `full_like`.

Whisper

XLS-R

XLSR-Wav2Vec2

### Multimodal Models

CLIP

**Data2Vec** "AttributeError: 'list' object has no attribute 'val'" in `reshape` op [is fixed now?]

**Data2VecAudio** The conversion completes without errors but the Core ML compiler cannot load the model.

Donut

FLAVA

**GroupViT** Conversion issue with `scatter_along_axis` operation.

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
