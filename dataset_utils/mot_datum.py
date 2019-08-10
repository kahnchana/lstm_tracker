import os

from dataset_utils.dataset_class import Dataset
from dataset_utils.datum import Datum
from dataset_utils.mot_obj import MOTObj
from collections import defaultdict


class MOTDatum(Datum):
    """
    Datum for Pascal VOC Segmentation. Contains bounding box data, class segmentation maps and instance segmentation
    maps.
    """

    def __init__(self, root_path, img_id, det, gt):
        self.det = det
        self.gt = gt
        self._gt_objects = None
        super(MOTDatum, self).__init__(root_path, img_id)

    @property
    def title(self):
        return self.img_id

    @property
    def im_path(self):
        return self._complete_path('{}/img1/{}')

    @property
    def objects(self):
        if self._objects is None:
            self._objects = [MOTObj.deserialize_gt(line.strip()) for line in self.gt]
        return self._objects

    @property
    def ground_truth(self):
        if self._gt_objects is None:
            self._gt_objects = [MOTObj.deserialize(line.strip()) for line in self.det]
        return self._gt_objects


class MOTSequence(Dataset):
    """
    Class holding datasets containing MOTDatum objects.
    """

    def __init__(self, root_path):
        """

        Args:
            root_path:      path to seq folder
        """
        _ids = sorted(os.listdir(os.path.join(root_path, 'img1')))
        self._all_img_ids = [x for x in _ids if x.endswith(".jpg") and not x.startswith(".")]
        self._det = defaultdict(list)
        self._gt = defaultdict(list)
        with open("{}/det/det.txt".format(root_path)) as fo:
            lines = fo.readlines()
            for line in lines:
                _temp = line.strip().split(',')
                if float(_temp[6]) > 0.5:
                    idx = int(_temp[0])
                    self._det[idx].append(line)

        if os.path.exists("{}/gt/gt.txt".format(root_path)):
            with open("{}/gt/gt.txt".format(root_path)) as fo:
                lines = fo.readlines()
                for line in lines:
                    _temp = line.strip().split(',')
                    idx = int(_temp[0])
                    self._gt[idx].append(line)

        super(MOTSequence, self).__init__(root_path, MOTDatum)

    @property
    def all_img_ids(self):
        return self._all_img_ids

    def create_datum(self, img_id: str) -> Datum:
        return self._datum_class(self.root_path, img_id, self._det[int(img_id[:-4])], self._gt[int(img_id[:-4])])


class MOTDataset:
    """
    Class for holding MOT sequences.
    """

    def __init__(self, root_path):
        """

        Args:
            root_path:      path to folder containing sequence sub-folders
        """
        self.root_path = root_path
        self.seq_folders = [x for x in os.listdir(self.root_path) if not x.startswith(".")]

    def get_sequence(self, seq_id):
        """

        Args:
            seq_id:     str (seq folder name)

        Returns:
            MOTSequence object instance
        """
        seq = MOTSequence(root_path=os.path.join(self.root_path, seq_id))
        return seq

    @property
    def sequences(self):
        return [self.get_sequence(seq_id) for seq_id in self.seq_folders]
