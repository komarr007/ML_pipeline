
import tensorflow as tf
import tensorflow_transform as tft
import diabetes_data_constant

_FEATURE_KEY = diabetes_data_constant.FEATURE_KEY
_LABEL_KEY = diabetes_data_constant.LABEL_KEY
_transformed_name = diabetes_data_constant.transformed_name

def preprocessing_fn(inputs):
  """Melakukan proses scaling z score pada feature"""
  
  outputs = {}
  for key in _FEATURE_KEY:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs
