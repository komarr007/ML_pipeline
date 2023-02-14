# %% [markdown]
# # Importing Library

# %%
import tensorflow as tf
import tfx
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import os

from tfx.dsl.components.common.resolver import Resolver 
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy 
from tfx.types import Channel 
from tfx.types.standard_artifacts import Model, ModelBlessing 

from tfx.proto import trainer_pb2

# %% [markdown]
# # Declaring Contants Variable

# %% [markdown]
# Code di bawah akan membuat variable konstan untuk pipeline

# %%
PIPELINE_NAME = "diabetes-pipeline"
SCHEMA_PIPELINE_NAME = "diabetes-tfdv-schema"

PIPELINE_ROOT = os.path.join('komarr007-pipeline', PIPELINE_NAME)

METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

# %%
DATA_ROOT = "data"

# %%
ctx = InteractiveContext(pipeline_root=PIPELINE_ROOT)

# %% [markdown]
# # Example Gen

# %% [markdown]
# code di bawah melakukan pembacaan data

# %%
example_gen = CsvExampleGen(input_base=DATA_ROOT)
ctx.run(example_gen)

# %% [markdown]
# # Statistic Gen

# %% [markdown]
# Code di bawah membuat statistic generator.

# %%
stats_gen = StatisticsGen(
    examples=example_gen.outputs['examples']
)

ctx.run(stats_gen)

# %%
ctx.show(stats_gen.outputs['statistics'])

# %% [markdown]
# # Schema Gen

# %% [markdown]
# Cde di bawah melakukan kontrol terhadap skema pada data

# %%
schema_gen = SchemaGen(statistics=stats_gen.outputs['statistics'])

ctx.run(schema_gen)

# %%
ctx.show(schema_gen.outputs['schema'])

# %% [markdown]
# # Anomalies Validator

# %% [markdown]
# Code di bawah akan mengecek anomali pada data

# %%
example_validator = ExampleValidator(
    statistics=stats_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
ctx.run(example_validator)

# %%
ctx.show(example_validator.outputs['anomalies'])

# %% [markdown]
# # Create Constant Module

# %% [markdown]
# code di bawah akan membuat contant module yang akan di gunakan untuk transformasi data

# %%
CONSTANT_MODULE_FILE = "diabetes_data_constant.py"

# %%
%%writefile {CONSTANT_MODULE_FILE}

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Outcome"
FEATURE_KEY = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


def transformed_name(key):
    """Rename transformed features"""

    return key + "_tn"

# %% [markdown]
# # Create Transform Module

# %% [markdown]
# Code di bawah membuat transformasi module

# %%
TRANSFORM_MODULE_FILE = "diabetes_data_transform.py"

# %%
%%writefile {TRANSFORM_MODULE_FILE}

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

# %% [markdown]
# # Transform pipeline

# %% [markdown]
# Code di bawah membuat transformasi untuk pipeline berdasarkan module transform yang telah dibuat

# %%
transform  = Transform(
    examples=example_gen.outputs['examples'],
    schema= schema_gen.outputs['schema'],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)
ctx.run(transform)

# %% [markdown]
# # Create Training Module

# %% [markdown]
# Code di bawah membuat training module yang berisikan model ML

# %%
TRAINER_MODULE_FILE = "diabetes_data_trainer.py"

# %%
%%writefile {TRAINER_MODULE_FILE}

import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft 
from tensorflow.keras import layers
import os  
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs
import diabetes_data_constant

_LABEL_KEY = diabetes_data_constant.LABEL_KEY
_FEATURE_KEY = diabetes_data_constant.FEATURE_KEY
_transformed_name = diabetes_data_constant.transformed_name

    
def get_model(show_summary: bool = True) -> tf.keras.models.Model:

    input_features = []

    for key in _FEATURE_KEY:
        input_features.append(
            tf.keras.Input(shape=(1,), name=_transformed_name(key))
        )

    x = tf.keras.layers.concatenate(input_features)
    x_1 = tf.keras.layers.Dense(16, activation="relu")(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x_1)

    inputs = input_features

    keras_model = tf.keras.models.Model(inputs, output)
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(),
        ],
    )
    if show_summary:
        keras_model.summary()

    return keras_model


def _gzip_reader_fn(filenames):
    """membaca gzip file dari input"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """fungsi untuk parsing tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """output untuk digunakan pada saat serving"""
        
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output, batch_size=64):

    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(_LABEL_KEY),
    )

    return dataset


def run_fn(fn_args):
    """main function"""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 64)

    print(train_dataset)
    
    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    callbacks = [tensorboard_callback]

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)

# %% [markdown]
# # Model Pipeline

# %% [markdown]
# Code di bawah akan mengeksekusi model pada module yang telah dibuat

# %%
from tfx.proto import trainer_pb2
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor

trainer = Trainer(
    module_file = os.path.abspath(TRAINER_MODULE_FILE),
    examples = transform.outputs['transformed_examples'],
    transform_graph = transform.outputs['transform_graph'],
    schema = schema_gen.outputs['schema'],
    train_args = trainer_pb2.TrainArgs(num_steps=100),
    eval_args = trainer_pb2.EvalArgs(num_steps=1)
)

ctx.run(trainer)

# %% [markdown]
# # Resolver

# %% [markdown]
# Code di bawah akan melakukan resolver

# %%
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy 
from tfx.types import Channel 
from tfx.types.standard_artifacts import Model, ModelBlessing 
 
model_resolver = Resolver(
    strategy_class= LatestBlessedModelStrategy,
    model = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')
 
ctx.run(model_resolver)

# %% [markdown]
# # Evaluator

# %% [markdown]
# Code di bawah akan mengevaluasi model yang telah dibuat

# %%
import tensorflow_model_analysis as tfma 
 
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='Outcome')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value':0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value':0.0001})
                    )
            )
        ])
    ]
)

# %%
from tfx.components import Evaluator
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
 
ctx.run(evaluator)

# %%
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)

# %% [markdown]
# # Pusher

# %% [markdown]
# Code dibawah membuat model siap untuk di serving

# %%
from tfx.components import Pusher 
from tfx.proto import pusher_pb2 
 
pusher = Pusher(
model=trainer.outputs['model'],
model_blessing=evaluator.outputs['blessing'],
push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory='serving_model_dir/diabetes-detection-model'))
)
 
ctx.run(pusher)

# %% [markdown]
# Code di bawah melakukan pembacaan terhadap data yang telah di transform yang akan digunakan untuk testing prediksi

# %%
# Get the URI of the output artifact representing the transformed examples, which is a directory
train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

for tfrecord in dataset.take(1):
  serialized_example = tfrecord.numpy()
  example = tf.train.Example()
  example.ParseFromString(serialized_example)
  print(example)

# %%
import requests
from pprint import PrettyPrinter
 
pp = PrettyPrinter()
pp.pprint(requests.get("https://mlpipeline-production.up.railway.app/v1/models/cc-model").json())


