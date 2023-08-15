August 15, 2023
def infer_sequence_length_from_config - made true
  hoping that if this is true, maybe the use_legacy_format may not need to be used.
  hoping that ct.EnumeratedShapes would kick in (maybe, maybe need to edit the convert.py file for that) would allow for caching/higher optimization also ANE usage
  ct.Shape is currently used, but was not converting llama models because RangeDim() upper_bounds is infinite, therefore erroring
  
def use_legacy_format - made true 
  is the work around if error is thrown
  need to see the intference type, tokens/sec
  if llama-7b produced by pcenq is using the legacy format, and it was 6.5 tokens/sec, then allowing for ct.EnumeratedShapes (if there is not more than 128 shapes) we would be set
  
May be able to add in config.py/convert.py pruning and palletization which will futher increase optimization for Apple devices and ANE.
