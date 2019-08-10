import numpy as np

_MATCHED = 0
_NO_MATCH = 1
_NEW = 2


class Track:
    """
    A single target track with state space `(x, y, h, w)` , where `(x, y)` is the center of the bounding box, `h` is
    the height, and 'w' is the width.

    Parameters
    ----------
    start_pos : ndarray
        Array of shape (4 + num_classes) with first four [cx, cy, h, w].
    num_classes: number of classes
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    iou_thresh: float
        IoU threshold for matching.
    time_steps: int
        Number of time steps to remember for a Track.

    Attributes
    ----------
    pos_data : ndarray
        Array of shape (time_steps, 4 + num_classes) with first four [cx, cy, h, w].
    track_id : int
        A unique track identifier.
    hits : int
        Total number of detections matching predictions from track (matches made for IoU > thresh).
    age : int
        Total number of frames since first occurrence.
    time_since_match : int
        Total number of frames since last match.

    """

    def __init__(self, start_pos, num_classes, track_id, iou_thresh=0.5, time_steps=10):
        self.pos_data = np.zeros(shape=(time_steps, 4 + num_classes))
        self.pos_data[-1, :] = start_pos
        self.track_id = track_id

        self.iou_thresh = iou_thresh
        self.hits = 1
        self.age = 1
        self.time_since_match = 0

        self.state = _NEW

    def to_cwh(self):
        """Get current position in bounding box centre-format `(centre x, centre y, height, width)`.

        Returns
        -------
        ndarray
            Array of shape (time_steps, 4 + num_classes).

        """
        return self.pos_data.copy()

    def to_tlbr(self):
        """Get current position in bounding box format `(min y, min x, max y, max x)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.pos_data.copy()
        _H, _W = ret[:, 3] / 2, ret[:, 2] / 2  # H,W are half height & width
        ret[:, 3], ret[:, 2] = ret[:, 0] + _W, ret[:, 1] + _H
        ret[:, 1], ret[:, 0] = ret[:, 0] - _W, ret[:, 1] - _H

        return ret

    def update(self, detection, prediction, iou=None):
        """Update internal parameters with a new detection

        Parameters
        ----------
        detection : np.array
            The associated detection as [cx, cy, h, w].
        prediction : np.array
            The associated prediction as [cx, cy, h, w].
        iou : float
            IoU value of detection with prediction (greater than thresh means it is a match)
        """
        self.age += 1

        if iou > self.iou_thresh:
            self.pos_data = np.r_[self.pos_data[1:, :], np.expand_dims(detection, axis=0)]
            self.hits += 1
            self.time_since_match = 0
            self.state = _MATCHED
        else:
            self.pos_data = np.r_[self.pos_data[1:, :], np.expand_dims(prediction, axis=0)]
            self.time_since_match += 1
            self.state = _NO_MATCH
