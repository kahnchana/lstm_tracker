from trainer.model import create_model
from trainer.helpers import to_bbox_tf
import tensorflow as tf
import numpy as np


class GraphRunnerOld:

    def __init__(self, graph_path=None):
        """
        Initialized GraphRunner instance. Creates the model, tf session, and asserts graph is correct
        """
        self.graph_path = graph_path
        self.model = create_model(timesteps=10, input_dim=4, num_classes=9, hidden_sizes=(32, 32))

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        sess.run(init)

        self.session = sess

        # assert input / output nodes in graph
        input_tensor = "input_tensor"
        output_tensors = ["x_comb/concat", "y_comb/concat", "h_comb/concat", "w_comb/concat"]
        all_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        assert input_tensor in all_nodes, "input tensor not found in graph"
        for output_tensor in output_tensors:
            assert output_tensor in all_nodes, "output tensor not found in graph"

    def restore(self, graph_path=None):
        # restore model
        if graph_path is not None:
            self.graph_path = graph_path
        assert self.graph_path is not None, "graph path has not been initialized"
        saver = tf.train.Saver()
        saver.restore(self.session, self.graph_path)

    def run(self, x):
        predictions = self.model(x, training=False)
        predictions = tf.stack(predictions, axis=-2)  # output is a tuple of tensors (x, y, h, w)
        y = to_bbox_tf(x=x, y=predictions, batch=True).eval(session=self.session)
        return y

    @staticmethod
    def prepare_inputs(x, time_steps=10):
        """
        Convert the inputs to the required format.
        Args:
            x:          inputs array (shape: time_steps, 4+num_classes)
            time_steps:  number of time-steps for model (default 10)

        Returns:
            prepared input array as np.array
        """
        x_time_steps, num_features = x.shape
        x = np.r_[np.zeros((time_steps - x_time_steps, num_features)), x]
        x = np.expand_dims(x, axis=0).astype(np.float32)

        return x
