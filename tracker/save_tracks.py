import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset_utils.kitti_datum import KITTIDataset
from dataset_utils.mot_datum import MOTDataset
from tracker.tracking import Tracker, _NO_MATCH
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

save_path = "{}/fyp_2019/LSTM_Kanchana/data/results/{}/{}".format(BASE_PATH, "KITTI", "{}.txt")

seq = dataset.sequences[6]

init = 1

with open(save_path.format(seq.seq_id), "a+") as fo:
    for frame_num, datum in tqdm(enumerate(seq.datums())):

        boxes = get_boxes_from_datum(dat=datum, kitti=True)

        if init:
            tracker.initiate_tracks(boxes)
            init = 0
            continue

        preds = tracker.predict()
        tracker.update(detections=boxes, predictions=preds)

        for track in tracker.tracks:

            if track.state == _NO_MATCH:
                continue

            tracks_pos = track.to_tlbr()[-1, :4]
            out_str = "{} {} {} {} {} {} \n".format(frame_num, track.track_id, *tracks_pos)
            # print(out_str)
            fo.write(out_str)
