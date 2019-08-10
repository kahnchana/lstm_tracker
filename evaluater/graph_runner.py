import numpy as np
import tensorflow as tf

from trainer.data import joint_data_gen
from trainer.model import model_fn

_PARAMS = {
    'Eval_IOU': 0.5,
    'output_bins': 5,
    'hidden layers': [32, 32],
    'num_classes': 9,
    'num_timesteps': 10
}


class GraphRunner:

    def __init__(self, base_path, experiment, checkpoint, model_params=_PARAMS):
        """
        Initialized GraphRunner instance. Creates the model, tf session, and asserts graph is correct
        """
        self.base_path = base_path
        self.model_dir = "{}/models/{}".format(self.base_path, experiment)
        self.checkpoint = checkpoint
        self.checkpoint_path = "{}/{}".format(self.model_dir, self.checkpoint)

        self.classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=self.model_dir,
            params=model_params
        )

    @staticmethod
    def prepare_inputs(x, y, time_steps=10, cut_off=5):
        """
        Convert the inputs to the required format.
        Args:
            x:              inputs array (shape: time_steps, 4+num_classes)
            y:              labels
            time_steps:     number of time-steps for model (default 10)
            cut_off:        limit for x-time steps

        Returns:
            prepared input array as np.array
        """
        x = x[cut_off:, :]
        x_time_steps, num_features = x.shape
        x = tf.concat([tf.zeros((time_steps - x_time_steps, num_features)), x], axis=0)
        return x, y

    def eval_input_fn(self):
        gen = tf.data.Dataset.from_generator(
            generator=joint_data_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape((10, 13)), tf.TensorShape((4, 10))),
            args=(["{}/data/kitti_tracks_{}.json".format(self.base_path, "{}"),
                   "{}/data/mot_tracks_{}.json".format(self.base_path, "{}")],
                  'val',
                  9,
                  True,
                  True)
        )
        gen = gen.map(self.prepare_inputs)
        gen = gen.repeat(count=1).batch(batch_size=128).prefetch(buffer_size=128 * 2)

        return gen

    def run_eval(self):
        results = self.classifier.evaluate(input_fn=self.eval_input_fn,
                                           checkpoint_path=self.checkpoint_path)
        return results

    def get_predictions(self, array):

        def pred_input_fn():
            features = array.astype(np.float32)
            return tf.convert_to_tensor(features), None

        predictions = self.classifier.predict(input_fn=pred_input_fn,
                                              checkpoint_path=self.checkpoint_path,
                                              yield_single_examples=False)
        return next(predictions)

    @staticmethod
    def prepare_inputs_np(x, time_steps=10):
        """
        Convert the inputs to the required format.
        Args:
            x:              inputs array (shape: time_steps, 4+num_classes)
            time_steps:     number of time-steps for model (default 10)

        Returns:
            prepared input array as np.array
        """
        x_time_steps, num_features = x.shape
        x = np.r_[np.zeros((time_steps - x_time_steps, num_features)), x]
        x = np.expand_dims(x, axis=0).astype(np.float32)

        return x
