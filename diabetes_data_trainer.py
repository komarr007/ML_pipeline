
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
