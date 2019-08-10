import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset_utils.kitti_datum import KITTIDataset
from dataset_utils.mot_datum import MOTDataset
from tracker.tracking import Tracker
from trainer.dataset_info import kitti_classes_reverse
from trainer.helpers import lbtr_to_chw, to_one_hot

from vis_utils.vis_datum import datum_with_labels

tracker = Tracker()


def get_boxes_from_datum(dat, kitti=True):
    i_w, i_h = dat.image.size

    if len(dat.objects) == 0:
        return np.zeros(shape=(0, 4 + 9))

    if kitti:
        _boxes = lbtr_to_chw(np.array([[x.x_min / i_w, x.y_min / i_h, x.x_max / i_w, x.y_max / i_h,
                                        kitti_classes_reverse[x.category]] for x in dat.objects]))
    else:
        _boxes = lbtr_to_chw(np.array([[x.x_min / i_w, x.y_min / i_h, x.x_max / i_w, x.y_max / i_h, 0]
                                       for x in dat.objects]))
    _boxes = to_one_hot(_boxes, 9)

    return _boxes


ious = []
pbar = tqdm()

BASE_PATH = "/Users/kanchana/Documents/current/FYP/"
dataset = KITTIDataset(root_path="{}/data/KITTI_tracking/data_tracking_image_2/training".format(BASE_PATH))
# dataset = MOTDataset(root_path="{}/data/MOT16/train".format(BASE_PATH))

seq = dataset.sequences[6]

init = 1

for i, datum in enumerate(seq.datums()):

    # if i < 235:
    #     continue

    boxes = get_boxes_from_datum(dat=datum, kitti=True)

    if init:
        tracker.initiate_tracks(boxes)
        init = 0
        continue

    preds = tracker.predict()
    tracker.update(detections=boxes, predictions=preds)
    im = datum.image.copy()
    tracker.draw_tracks(im)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.imshow(np.array(im))
    ax1.axis("off")

    im_orig = datum_with_labels(datum)
    ax2.imshow(np.array(im_orig))
    ax2.axis("off")

    # plt.waitforbuttonpress()
    plt.pause(0.1)
    plt.close()

    # from vis_utils.vis_datum import datum_with_labels
    # datum_with_labels(datum).show()
