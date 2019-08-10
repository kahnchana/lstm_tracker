import motmetrics as mm
import numpy as np

from dataset_utils.kitti_datum import KITTIDataset
from dataset_utils.track_datum import TrackObjHandler

BASE_PATH = "/Users/kanchana/Documents/current/FYP/"

dataset = KITTIDataset(root_path="{}/data/KITTI_tracking/data_tracking_image_2/training".format(BASE_PATH))
# dataset = MOTDataset(root_path="{}/data/MOT16/train".format(BASE_PATH))

save_path = "{}/fyp_2019/LSTM_Kanchana/data/results/{}/{}".format(BASE_PATH, "KITTI", "{}.txt")

track_seq = TrackObjHandler("{}/fyp_2019/LSTM_Kanchana/data/results/KITTI".format(BASE_PATH), "0006")
gt_seq = dataset.sequences[6]

acc = mm.MOTAccumulator(auto_id=True)

for i, gt in enumerate(gt_seq.datums()):

    if i == 0:
        continue

    i_w, i_h = gt.image.size
    gt_bb = np.array([np.array([x.x_min / i_w, x.y_min / i_h,
                                (x.x_max - x.x_min) / i_w, (x.y_max - x.y_min) / i_h]) for x in gt.objects])

    track = track_seq.tracks[int(gt.img_id[:-4])]
    t_bb = np.array([np.array([x.x_min, x.y_min, x.x_max - x.x_min, x.y_max - x.y_min]) for x in track])

    gt_ids = np.array([x.track for x in gt.objects])
    track_ids = np.array([x.track for x in track])
    iou_dist = mm.distances.iou_matrix(gt_bb, t_bb, max_iou=1)  # x_min, y_min, W, H
    acc.update(gt_ids, track_ids, iou_dist)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'mostly_tracked', 'mostly_lost',
                                   'num_switches', 'num_matches', 'num_objects'], name='acc')
pretty_summary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(pretty_summary)
