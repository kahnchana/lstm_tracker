import tensorflow as tf
import numpy as np

from trainer.data import joint_data_gen
from trainer.model import model_fn

BASE_PATH = "/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana"
model_dir = "{}/models/exp02".format(BASE_PATH)


classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    params={
        'Eval_IOU': 0.5,
        'output_bins': 5,
        'hidden layers': [32, 32],
        'num_classes': 9,
        'num_timesteps': 10
    }
)


def prepare_inputs(x, y, time_steps=10, cut_off=8):
    """
    Convert the inputs to the required format.
    Args:
        x:              inputs array (shape: time_steps, 4+num_classes)
        time_steps:     number of time-steps for model (default 10)
        cut_off:        limit for x-time steps

    Returns:
        prepared input array as np.array
    """
    x = x[cut_off:, :]
    x_time_steps, num_features = x.shape
    x = tf.concat([tf.zeros((time_steps - x_time_steps, num_features)), x], axis=0)
    return x, y


def eval_input_fn():
    gen = tf.data.Dataset.from_generator(
        generator=joint_data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((10, 13)), tf.TensorShape((4, 10))),
        args=(["{}/data/kitti_tracks_{}.json".format(BASE_PATH, "{}"),
               "{}/data/mot_tracks_{}.json".format(BASE_PATH, "{}")],
              'val',
              9,
              True,
              True)
    )
    gen = gen.map(prepare_inputs)
    gen = gen.repeat(count=1).batch(batch_size=128).prefetch(buffer_size=128 * 2)

    return gen


def pred_input_fn():
    features = np.ones(shape=(1, 10, 13), dtype=np.float32)
    return tf.convert_to_tensor(features), None


predictions = classifier.predict(input_fn=pred_input_fn,
                                 checkpoint_path="{}/model.ckpt-109169".format(model_dir),
                                 yield_single_examples=True)

results = classifier.evaluate(input_fn=eval_input_fn,
                              checkpoint_path="{}/model.ckpt-109169".format(model_dir))

