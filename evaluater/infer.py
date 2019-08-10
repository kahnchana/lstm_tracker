import argparse

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from trainer.data import mot_data_gen, kitti_data_gen
from trainer.helpers import bbox_overlap_iou_np, to_bbox_tf
from trainer.model import create_model
from vis_utils.vis_datum import ImageBoxes


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='train_network')
    parser.add_argument('--dataset', dest='dataset', help='KITTI / MOT', default='KITTI')
    parser.add_argument('--data_path', dest='data_path', help='path to data JSON',
                        default="/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana/data/kitti_tracks.json")
    parser.add_argument('--graph', dest='graph', help='model output directory',
                        default="/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana/models/exp01")
    arguments = parser.parse_args()
    return arguments


def main(graph_file_path="/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana/models/exp04/"):
    graph_file_path = tf.train.latest_checkpoint(graph_file_path)
    model = create_model(timesteps=10, input_dim=4, num_classes=9, hidden_sizes=(32, 32))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)
    # restore model
    saver = tf.train.Saver()
    saver.restore(sess, graph_file_path)

    input_tensor = "input_tensor"
    output_tensors = ["x_comb/concat", "y_comb/concat", "h_comb/concat", "w_comb/concat"]

    # assert input / output nodes in graph
    all_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    assert input_tensor in all_nodes, "input tensor not found in graph"
    for output_tensor in output_tensors:
        assert output_tensor in all_nodes, "output tensor not found in graph"

    auto_time = True
    gen = mot_data_gen(split='val', testing=True, one_hot_classes=True, anchors=True)

    while True:
        pick = np.random.choice(range(10))
        if pick == 0:
            gen = mot_data_gen(split='val', testing=True, one_hot_classes=True, anchors=True)
        elif pick == 1:
            gen = kitti_data_gen(split='val', testing=True, one_hot_classes=True, anchors=True)
        else:
            pass

        # load data
        x, y, x_im, y_im = next(gen)

        # obtain predictions
        predictions = model(np.expand_dims(x, axis=0).astype(np.float32), training=False)
        predictions = tf.stack(predictions, axis=-2)
        y_pred = to_bbox_tf(x=np.expand_dims(x, axis=0).astype(np.float32), y=predictions,
                            batch=True).eval(session=sess)
        y = to_bbox_tf(x=np.expand_dims(x, axis=0).astype(np.float32),
                       y=np.expand_dims(y, axis=0).astype(np.float32),
                       batch=True).eval(session=sess)
        iou = bbox_overlap_iou_np(y_pred[:, :4], y[:, :4], ar=False)
        print(iou, np.greater_equal(iou, 0.5))

        fig, ax = plt.subplots(1)
        image = ImageBoxes(path=y_im)
        plt.axis("off")
        for num, i in enumerate(x):
            im = plt.imread(x_im[num])
            ih, iw, _ = im.shape
            cx, cy, h, w = i[:4]
            image.add_box([cx, cy, w, h], color='blue')
        for i in [y[0]]:
            im = plt.imread(y_im)
            ih, iw, _ = im.shape
            cx, cy, h, w = i[:4]
            image.add_box([cx, cy, w, h], color='red', thickness=4)

        cx, cy, h, w = y_pred[0, :4]
        image.add_box([cx, cy, w, h], color='orange', thickness=4)

        ax.imshow(np.array(image.get_final()))
        if auto_time:
            plt.pause(0.1)
        else:
            plt.waitforbuttonpress()
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    main(graph_file_path=args.graph)
