import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from trainer.data import kitti_data_gen, mot_data_gen
from vis_utils.vis_datum import ImageBoxes

_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'mean_squared_loss',
                                        'train_accuracy'])


def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None):
    """Function to get LoggingTensorHook.
    Args:
      every_n_iter: `int`, print the values of `tensors` once every N local
        steps taken on the current worker.
      tensors_to_log: List of tensor names or dictionary mapping labels to tensor
        names. If not set, log _TENSORS_TO_LOG by default.
    Returns:
      Returns a LoggingTensorHook with a standard set of tensors that will be
      printed to stdout.
    """
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG

    return tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=every_n_iter)


def bbox_cross_overlap_iou(bboxes1, bboxes2):
    """
    Coordinates are normalized float values between (0,1).

    Args:
        bboxes1:    shape (total_bboxes1, 4) with center_x, center_y, w, h point order.
        bboxes2:    shape (total_bboxes2, 4) with center_x, center_y, w, h point order.

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2) comparing each bbox in list one to each in list two. The IoU
        (intersection over union) of bboxes1[i] and bboxes2[j] are in [i, j] of output matrix tensor.
    """

    x1, y1, w1, h1 = tf.split(bboxes1, 4, axis=1)
    x2, y2, w2, h2 = tf.split(bboxes2, 4, axis=1)

    xi1 = tf.maximum(x1, tf.transpose(x2))
    xi2 = tf.minimum(x1, tf.transpose(x2))

    yi1 = tf.maximum(y1, tf.transpose(y2))
    yi2 = tf.minimum(y1, tf.transpose(y2))

    wi = w1 / 2.0 + tf.transpose(w2 / 2.0)
    hi = h1 / 2.0 + tf.transpose(h2 / 2.0)

    inter_area = tf.maximum(wi - (xi1 - xi2), 0) * tf.maximum(hi - (yi1 - yi2), 0)

    bboxes1_area = w1 * h1
    bboxes2_area = w2 * h2

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    return inter_area / (union + 0.0001)


def bbox_cross_overlap_iou_np(bboxes1, bboxes2):
    """
    Coordinates are normalized float values between (0,1).

    Args:
        bboxes1:    shape (total_bboxes1, 4) with center_x, center_y, h, w point order.
        bboxes2:    shape (total_bboxes2, 4) with center_x, center_y, h, w point order.

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2) comparing each bbox in list one to each in list two. The IoU
        (intersection over union) of bboxes1[i] and bboxes2[j] are in [i, j] of output matrix tensor.
    """

    x1, y1, h1, w1 = np.split(bboxes1, 4, axis=1)
    x2, y2, h2, w2 = np.split(bboxes2, 4, axis=1)

    xi1 = np.maximum(x1, np.transpose(x2))
    xi2 = np.minimum(x1, np.transpose(x2))

    yi1 = np.maximum(y1, np.transpose(y2))
    yi2 = np.minimum(y1, np.transpose(y2))

    wi = w1 / 2.0 + np.transpose(w2 / 2.0)
    hi = h1 / 2.0 + np.transpose(h2 / 2.0)

    inter_area = np.maximum(wi - (xi1 - xi2), 0) * np.maximum(hi - (yi1 - yi2), 0)

    bboxes1_area = w1 * h1
    bboxes2_area = w2 * h2

    union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

    return inter_area / (union + 0.0001)

def bbox_overlap_iou(bboxes1, bboxes2, ar=False, iou_thresh=False):
    """
    Coordinates are normalized float values between (0,1). Calculates one to one IOU only.
    It is required for total_bboxes1 = total_bboxes2.

    Args:
        bboxes1:    shape (total_bboxes1, 4) with center_x, center_y, h, w point order.
        bboxes2:    shape (total_bboxes2, 4) with center_x, center_y, h, w point order.
        ar:         if True, ar (=h/w) taken instead of w.
        iou_thresh: if True, results bool array output

    Returns:
        Tensor with shape (total_bboxes1,) with the IoU (intersection over union) between bboxes1[i] and bboxes2[i] in
        position [i, j] of output vector tensor.
    """

    x1, y1, h1, w1 = tf.split(bboxes1, 4, axis=1)
    x2, y2, h2, w2 = tf.split(bboxes2, 4, axis=1)
    if ar:
        w1 = h1 / w1
        w2 = h2 / w2

    xi1 = tf.maximum(x1, x2)
    xi2 = tf.minimum(x1, x2)

    yi1 = tf.maximum(y1, y2)
    yi2 = tf.minimum(y1, y2)

    wi = w1 / 2.0 + w2 / 2.0
    hi = h1 / 2.0 + h2 / 2.0

    inter_area = tf.maximum(wi - (xi1 - xi2), 0) * tf.maximum(hi - (yi1 - yi2), 0)

    bboxes1_area = w1 * h1
    bboxes2_area = w2 * h2

    union = (bboxes1_area + bboxes2_area) - inter_area
    iou = inter_area / (union + 0.0001)
    if iou_thresh:
        iou = tf.cast(tf.greater_equal(iou, iou_thresh), tf.int8)
    return iou


def bbox_overlap_iou_np(bboxes1, bboxes2, ar=False, iou_thresh=False):
    """
    Coordinates are normalized float values between (0,1). Calculates one to one IOU only.
    It is required for total_bboxes1 = total_bboxes2.

    Args:
        bboxes1:    shape (total_bboxes1, 4) with center_x, center_y, h, w point order.
        bboxes2:    shape (total_bboxes2, 4) with center_x, center_y, h, w point order.
        ar:         if True, ar (=h/w) taken instead of w.

    Returns:
        Tensor with shape (total_bboxes1,) with the IoU (intersection over union) between bboxes1[i] and bboxes2[i] in
        position [i, j] of output vector tensor.
    """

    x1, y1, h1, w1 = np.split(bboxes1, 4, axis=1)
    x2, y2, h2, w2 = np.split(bboxes2, 4, axis=1)
    if ar:
        w1 = h1 / w1
        w2 = h2 / w2

    xi1 = np.maximum(x1, x2)
    xi2 = np.minimum(x1, x2)

    yi1 = np.maximum(y1, y2)
    yi2 = np.minimum(y1, y2)

    wi = w1 / 2.0 + w2 / 2.0
    hi = h1 / 2.0 + h2 / 2.0

    inter_area = np.maximum(wi - (xi1 - xi2), 0) * np.maximum(hi - (yi1 - yi2), 0)

    bboxes1_area = w1 * h1
    bboxes2_area = w2 * h2

    union = (bboxes1_area + bboxes2_area) - inter_area
    iou = inter_area / (union + 0.0001)
    if iou_thresh:
        iou = np.greater_equal(iou, iou_thresh)
    return iou


def chw_to_lbtr(box, ar=False):
    x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    if ar:
        w = h / w
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    return [y_min, x_min, y_max, x_max]


def lbtr_to_chw(box):
    out_box = box.copy()
    x_min, y_min, x_max, y_max = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    out_box[:, 0] = (x_min + x_max) / 2
    out_box[:, 1] = (y_min + y_max) / 2
    out_box[:, 2] = x_max - x_min
    out_box[:, 3] = y_max - y_min

    return out_box


def to_one_hot(x, num_classes):
    temp = np.zeros(shape=(x.shape[0], num_classes))
    temp[np.array(range(len(x[:, 4]))), x[:, 4].astype(int)] = 1
    x = np.concatenate([x[:, :4], temp.astype(float)], axis=-1)
    assert x.shape[1] == 4 + num_classes, "wrong shape for x"

    return x


def to_bbox_tf(x, y, batch=True, bins=(-0.5, 0, 0.1, 0.2, 0.5)):
    """
    Convert anchor-bin representation to normalized coordinate representation
    Args:
        x:          generator data output
        y:          generator label output
        batch:      is batch dimension present
        bins:       bins used

    Returns:

    """
    num_bins = len(bins)
    bins = tf.constant(bins, dtype=tf.float32)
    function_to_map = lambda b: bins[b]
    if batch:
        idx = tf.argmax(y[:, :, :num_bins], axis=-1, output_type=tf.int32)
        dist = tf.reduce_sum(y[:, :, num_bins:] * tf.one_hot(idx, num_bins), axis=-1)
        bin_vals = tf.map_fn(fn=lambda a: tf.map_fn(function_to_map, a, dtype=tf.float32), elems=idx, dtype=tf.float32)
        real_dist = tf.cast(bin_vals, dtype=tf.float32) + tf.cast(dist, dtype=tf.float32)
        y = real_dist + x[:, -1, :4]
    else:
        idx = tf.argmax(y[:, :num_bins], axis=-1)
        dist = y[:, num_bins:][range(idx.shape[0]), idx]
        bin_vals = tf.map_fn(fn=function_to_map, elems=idx, dtype=tf.int32)
        real_dist = tf.cast(bin_vals, dtype=tf.float32) + dist
        y = real_dist + x[-1, :4]

    return y


class SaveImages(tf.train.SessionRunHook):

    def __init__(self, model, dataset, count=10):
        self.model = model
        if dataset == "KITTI":
            self.gen = kitti_data_gen
        else:
            self.gen = mot_data_gen
        self.y_pred = []
        self.data = []
        self.count = count

    def before_run(self, run_context):
        for i in range(self.count):
            gen = self.gen(split='val', testing=True)
            x, y, x_im, y_im = next(gen)
            self.data.append([x, y, x_im, y_im])
            self.y_pred.append(self.model(np.expand_dims(x, axis=0), training=False))

    def after_run(self, run_context, run_values):
        for j in range(self.count):
            x, y, x_im, y_im = self.data[j]
            image = ImageBoxes(path=y_im)
            plt.axis("off")
            for num, i in enumerate(x):
                im = plt.imread(x_im[num])
                ih, iw, _ = im.shape
                cx, cy, h, ar, cl = i
                image.add_box([cx, cy, h / ar, h], color='blue')
            for i in [y]:
                im = plt.imread(y_im)
                ih, iw, _ = im.shape
                cx, cy, h, ar, cl = i
                image.add_box([cx, cy, h / ar, h], color='red')

            cx, cy, h, ar, _ = self.y_pred[j][0]
            image.add_box([cx, cy, h / ar, h], color='green')

            tf.summary.image(np.expand_dims(np.array(image.get_final()), axis=0), name="image_{}".format(j))


if __name__ == '__main__':
    tf.enable_eager_execution()

    bboxes1_val = np.array([[0.70775015, 0.4647078, 0.09439638, 1.8295372],
                            [0.71460492, 0.46425386, 0.09746107, 1.804698],
                            [0.72189783, 0.4637695, 0.10073144, 1.77913711],
                            [0.72967244, 0.46325153, 0.1042289, 1.75281998]])

    bboxes2_val = np.array([[0.73856634, 0.46062235, 0.10825005, 1.72384472],
                            [0.74709516, 0.45784283, 0.11233135, 1.69729068],
                            [0.72967244, 0.46325153, 0.1042289, 1.75281998],
                            [0.76612634, 0.45160367, 0.12149236, 1.64145537]])

    iou_val = bbox_overlap_iou(bboxes1_val, bboxes2_val)
    print(iou_val)
