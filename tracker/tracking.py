import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluater.graph_runner import GraphRunner
from tracker.track import Track, _NO_MATCH
from trainer.helpers import bbox_cross_overlap_iou_np
from vis_utils.vis_utils import draw_bounding_box_on_image

LSTM_INFO = (
    "/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana",
    "exp02",
    "model.ckpt-109169"
)


class Tracker:
    """
    This is the multi-target tracker.

    Parameter
    ----------
    lstm_info : tuple
        Info for loading trained LSTM model.
    min_iou_distance : float
        IoU threshold.
    num_classes: int
        Number of object classes
    time_steps: int
        Number of time steps

    Attributes
    ----------
    min_iou_distance : float
        IoU matching threshold.
    predictor : evaluater.graph_runner.GraphRunner
        LSTM graph for predicting next track.
    tracks : List[Track]
        The list of active tracks at the current time step.
    num_classes: int
        Number of object classes
    time_steps: int
        Number of time steps
    """

    def __init__(self, lstm_info=LSTM_INFO, min_iou_distance=0.5, num_classes=9, time_steps=10, max_no_hit=6):

        self.predictor = GraphRunner(
            base_path=lstm_info[0],
            experiment=lstm_info[1],
            checkpoint=lstm_info[2])
        self.tracks = []
        self.min_iou_distance = min_iou_distance
        self.num_classes = num_classes
        self.time_steps = time_steps

        self._next_id = 1
        self.max_no_hit = max_no_hit

    def predict(self):
        """Obtain the next prediction from each track. Returns an array of shape (num_tracks, 4 + num_classes).
        """
        out_array = []
        if len(self.tracks) < 1:
            return np.zeros(0, 4 + self.num_classes)
        for track in self.tracks:
            out_array.append(track.to_cwh())
        out_array = np.array(out_array)
        predictions = self.predictor.get_predictions(out_array)

        return np.concatenate([predictions, out_array[:, -1, 4:]], axis=-1)

    def update(self, detections, predictions):
        """Perform updates on tracks.

        Parameters
        ----------
        detections : np.array
            Array of detections at the current time step of shape (count, 4 + num_classes).
        predictions : np.array
            Array of predictions at the current time step of shape (count, 4 + num_classes).
        """
        num_detections = detections.shape[0]
        num_predictions = predictions.shape[0]

        unmatched_detections = set()
        unmatched_tracks = set()

        # Get matches.
        iou_matrix = bbox_cross_overlap_iou_np(detections[:, :4], predictions[:, :4])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Find valid matches and update tracks
        for det_idx, pred_idx in zip(row_ind, col_ind):
            iou = iou_matrix[det_idx, pred_idx]
            if iou > self.min_iou_distance:
                self.tracks[pred_idx].update(detections[det_idx], predictions[pred_idx], iou)
            else:
                unmatched_tracks.add(pred_idx)
                unmatched_detections.add(det_idx)

        # find non-matches
        for det_idx in range(num_detections):
            if det_idx not in row_ind:
                unmatched_detections.add(det_idx)

        for pred_idx in range(num_predictions):
            if pred_idx not in col_ind:
                unmatched_tracks.add(pred_idx)

        # update unmatched tracks
        for pred_idx in unmatched_tracks:
            prediction = predictions[pred_idx]
            self.tracks[pred_idx].update(None, prediction, 0)

        # initiate new tracks
        for det_idx in unmatched_detections:
            track = Track(start_pos=detections[det_idx], num_classes=self.num_classes, track_id=self._next_id,
                          iou_thresh=self.min_iou_distance, time_steps=self.time_steps)
            self.tracks.append(track)
            self._next_id += 1

        # terminate older tracks
        tracks_to_remove = []
        for i in range(len(self.tracks)):
            if self.tracks[i].time_since_match > self.max_no_hit:
                tracks_to_remove.append(i)
        for i in tracks_to_remove:
            _ = self.tracks.pop(i)

    def initiate_tracks(self, detections):
        # initiate new tracks
        for detection in detections:
            track = Track(start_pos=detection, num_classes=self.num_classes, track_id=self._next_id,
                          iou_thresh=self.min_iou_distance, time_steps=self.time_steps)
            self.tracks.append(track)
            self._next_id += 1

    def draw_tracks(self, image):
        """
        Draws bounding boxes (for each track) on given image
        Args:
            image:  PIL image instance

        Returns:
            None. Draws in-place.
        """

        for track in self.tracks:

            if track.state == _NO_MATCH:
                continue

            box = track.to_tlbr()[-1, :4]  # ymin, xmin, ymax, xmax
            assert box.shape == (4,), "invalid shape: {}".format(box.shape)

            draw_bounding_box_on_image(image, box[0], box[1], box[2], box[3],
                                       display_str_list=["{:02d}".format(track.track_id)])
