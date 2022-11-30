# Design notes for Core ML exporters

The design of the Core ML exporter for 🤗 Transformers is based on that of the ONNX exporter. Both are used in the same manner and in some places the code is very similar. However, there are also differences due to the way Core ML works. This file documents the decisions that went into building the Core ML exporter.

## Philosophy

An important goal of Core ML is to make using models completely hands-off. For example, if a model requires an image as input, you can simply give it an image object without having to preprocess the image first. And if the model is a classifier, the output is the winning class label instead of a logits tensor. The Core ML exporter will add extra operations to the beginning and end of the model where possible, so that users of these models do not have to do their own pre- and postprocessing if Core ML can already handle this for them.

The Core ML exporter is built on top of `coremltools`. This library first converts the PyTorch or TensorFlow model into an intermediate representation known as MIL, then performs optimizations on the MIL graph, and finally serializes the result into a `.mlmodel` or `.mlpackage` file (the latter being the preferred format).

Design of the exporter:

- The Core ML conversion process is described by a `CoreMLConfig` object, analogous to `OnnxConfig`.

- In order to distinguish between the `default` task for text models and vision models, the config object must have a `modality` property. Unfortunately, there is no way determine the modality from the `AutoModel` object, so this property must be set in the `CoreMLConfig` subclass.

- The standard `CoreMLConfig` object already chooses appropriate input and output descriptions for most models. Only models that do something different, for example use BGR input images instead of RGB, need to have their own config object.

- If a user wants to change properties of the inputs or outputs (name, description, sequence length, other settings), they have to subclass the `XYZCoreMLConfig` object and override these methods. Not very convenient, but it's also not something people will need to do a lot — and if they do, it means we made the wrong default choice.

- Where possible, the behavior of the converted model is described by the tokenizer or feature extractor. For example, to use a different input image size, the user would need to create the feature extractor with those settings and use that during the conversion instead of the default feature extractor.

- The `FeaturesManager` code is copied from `transformers.onnx.features` with minimal changes to the logic, only the table with supported models is different (and using `CoreMLConfig` instead of `OnnxConfig`).

Extra stuff the Core ML exporter does:

- For image inputs, mean/std normalization is performed by the Core ML model. Resizing and cropping the image still needs to be done by the user but is usually left to other Apple frameworks such as Vision.

- Tensor inputs may have a different datatype. Specifically, `bool` and `int64` are converted to `int32` inputs, as that is the only integer datatype Core ML can handle.

- Classifier models that output a single prediction for each input example are treated as special by Core ML. These models have two outputs: one with the class label of the best prediction, and another with a dictionary giving the probabilities for all the classes.

- Models that perform classification but do not fit into Core ML's definition of a classifier, for example a semantic segmentation model, have the list of class names added to the model's metadata. Core ML ignores these class names but they can be retrieved by writing a few lines of Swift code.

- Because the goal is to make the converted models as convenient as possible for users, any model that predicts `logits` has the option of applying a softmax, to output probabilities instead of logits. This option is enabled by default for such models. For image segmentation models, there can be two operations inserted: upsampling to the image's original spatial dimensions, followed by an argmax to select the class index for each pixel.

- The exporter may add extra metadata to allow making predictions from Xcode's model previewer.

- Quantization and other optimizations can automatically be applied by `coremltools`, and therefore are part of the Core ML exporting workflow. The user can always make additional changes to the Core ML afterwards using `coremltools`, such as renaming the inputs and outputs, applying quantization, etc.

Note: Tokenizers are not a built-in feature of Core ML. A model that requires tokenized input must be tokenized by the user themselves. This is outside the scope of the Core ML exporter.

## Supported tasks

The Core ML exporter supports most of the tasks that the ONNX exporter supports, except for:

- `image-segmentation` / `AutoModelForImageSegmentation`

Tasks that the Core ML exporter supports but the ONNX exporter currently doesn't:

- `next-sentence-prediction`
- `semantic-segmentation`

Tasks that neither of them support right now:

- `AutoModelForAudioClassification`
- `AutoModelForAudioFrameClassification`
- `AutoModelForAudioXVector`
- `AutoModelForCTC`
- `AutoModelForInstanceSegmentation`
- `AutoModelForPreTraining`
- `AutoModelForSpeechSeq2Seq`
- `AutoModelForTableQuestionAnswering`
- `AutoModelForVideoClassification`
- `AutoModelForVision2Seq`
- `AutoModelForVisualQuestionAnswering`
- `...DoubleHeadsModel`
- `...ForImageClassificationWithTeacher`

Tasks that could be improved:

- `object-detection`. If a Core ML model outputs the predicted bounding boxes in a certain manner, the user does not have to do any decoding and can directly use these outputs in their app (through the Vision framework). Currently, the Core ML exporter does not add this extra functionality.

## Missing features

The following are not supported yet but would be useful to add:

- Flexible input sizes. Core ML models typically work with fixed input dimensions, but it also supports flexible image sizes and tensor shapes. The exporter currently supports flexible sequence lengths, but not image sizes.

    - Note: Certain models, notably BERT, currently give conversion errors with a flexible sequence length. This appears to be an issue with coremltools.

- More quantization options. coremltools 6 adds new quantization options for ML Program models, plus options for sparsifying weights.

- `validate_model_outputs`: If the model supports a flexible input sequence length, run the test three times: once with the maximum length (that's what happens now), once with the minimum length, and once with a length in between (possibly randomly chosen).

There are certain models that cannot be converted because of the way they are structured, or due to limitations and bugs in coremltools. Sometimes these can be fixed by making changes to the Transformers code, by implementing missing ops, or by filing bugs against coremltools. Trying to get as many Transformers models to export without issues is a work in progress.

### `-with-past` versions for seq2seq models

The encoder portion of the model is easy: this does not have a `past_key_values` option, so this is always converted with `use_past=False`.

When the decoder is used with `use_cache=True`, it needs to accept a `past_key_values` tensor that consists of a 4-tuple for each layer with the key/value for the decoder but also the key/value for the encoder. The decoder and encoder tensors have different shapes because they have different sequence lengths.

The encoder past key/values only need to be computed once, on the first iteration, and then they're simply re-used by the model on subsequent iterations. The decoder past key/values tensors grow in size with each iteration.

Handling the decoder past key/values tensors in Core ML is not a problem. On the first iteration, you can pass in a tensor with a shape of `(batch, num_layers, 0, num_heads)` or just leave out this tensor completely as it is marked optional. The model returns a new past key/values tensor and you simply pass that in on the next iteration.

This does not work for the encoder key/values. Core ML cannot perform branching logic in the model (not entirely true but its branching operation involves running a submodel and is rather complicated) and so the JIT trace must always choose one of the paths.

What this means is: If we specify dummy encoder key/value inputs during the JIT trace, then the cross-attention layer will not perform the `k_proj` and `v_proj` operations on the encoder's hidden state outputs.

In `BartAttention` that is these lines:

```python
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            ...
```

Here, `past_key_value` is the encoder key/values tensors and `key_value_states` is the encoder's last hidden state. The Core ML model can only include one of these branches, not both.

If during the JIT trace we pass in dummy tensors for the encoder key/value tensors, then the first branch is taken and `k_proj` and `v_proj` are never executed. The problem is that we need those projection operations to happen on the very first iteration.

In theory, we could solve this by never using the encoder key/values tensors, so that the second branch is always taken. This is less efficient, since it involves performing the same linear layers over and over, but at least it will work.

However, this workaround fails when an encoder attention mask is provided. In `BartDecoderLayer` the following happens:

```python
cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
```

Since the `past_key_value` tensor is now a 2-tuple instead of a 4-tuple (since we're no longer providing the encoder key/values), the expression `past_key_value[-2:]` will attempt to use the decoder key/values tensors for the cross attention. It should use the tensors at indices 2 and 3, but because the tuple only has two tensors in it now, this will use indices 0 and 1 — which are not the correct tensors!

Since the key/values from indices 0,1 have the target sequence length from the decoder, the encoder's `attention_mask` cannot be applied.

And even if we don't use this attention mask, what happens is incorrect anyway. The second branch will still never be taken (as `cross_attn_past_key_value` is not None) and `k_proj` and `v_proj` are never executed.

I currently don't see a solution to this except perhaps rewriting the decoder layer to do the following instead, but that requires changing a lot of source files in `transformers` and is a suboptimal solution anyway.

```python
cross_attn_past_key_value = past_key_value[-2:] if (past_key_value is not None and len(past_key_value) > 2) else None
```

We could also export two versions of the decoder model: one for the first iteration and one for the remaining iterations but that's not great either.

## Assumptions made by the exporter

The Core ML exporter needs to make certain assumptions about the Transformers models. These are:

- A vision `AutoModel` is expected to output hidden states. If there is a second output, this is assumed to be from the pooling layer.

- The input size for a vision model is given by the feature extractor's `crop_size` property if it exists and `do_center_crop` is true, or otherwise by its `size` property.

- The image normalization for a vision model is given by the feature extractor's `image_std` and `image_mean` if it has those, otherwise assume `std = 1/255` and `mean = 0`.

- The `masked-im` task expects a `bool_masked_pos` tensor as the second input. If `bool_masked_pos` is provided, some of these models return the loss value and others don't. If more than one tensor is returned, we assume the first one is the loss and ignore it.

- If text models have two inputs, the second one is the `attention_mask`. If they have three inputs, the third is `token_type_ids`.

- The `object-detection` task outputs logits and boxes (only tested with YOLOS so far).

- If bicubic resizing is used, it gets replaced by bilinear since Core ML doesn't support bicubic. This has a noticeable effect on the predictions, but usually the model is still usable.

## Other remarks

- Just as in the ONNX exporter, the `validate_model_outputs()` function takes an `atol` argument for the absolute tolerance. It might be more appropriate to do this test as `max(abs(coreml - reference)) / max(abs(reference))` to get an error measurement that's relative to the magnitude of the values in the output tensors.

- Image classifier models have the usual `classLabel` and `probabilities` outputs, but also a "hidden" `var_xxx` output with the softmax results. This appears to be a minor bug in the converter; it doesn't hurt anything to keep this extra output.

## Running the tests

The unit tests attempt to convert all supported models, and verify that their output is close to that of the original models. This can be very slow! These tests require a Mac.

```
$ cd exporters
$ RUN_SLOW=1 pytest tests/test_coreml.py --capture=sys -W ignore
```

The `--capture=sys` and `-W ignore` arguments are used to suppress the coremltools progress bars and other messages.

Tip: After running the tests, go into `/private/var/folders/...` and remove all the `.mlpackage` and `.mlmodel` files, as well as the `com.apple.MetalPerformanceShadersGraph` directory. coremtools leaves a lot of junk here that can quickly eat up your local storage space.
