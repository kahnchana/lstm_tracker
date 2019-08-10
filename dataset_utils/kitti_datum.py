import os

from dataset_utils.dataset_class import Dataset
from dataset_utils.datum import Datum
from dataset_utils.kitti_obj import KITTIObj
from collections import defaultdict


class KITTIDatum(Datum):
    """
    Datum for Pascal VOC Segmentation. Contains bounding box data, class segmentation maps and instance segmentation
    maps.
    """

    def __init__(self, root_path, img_id, gt):
        self.gt = gt
        super(KITTIDatum, self).__init__(root_path, img_id)

    @property
    def title(self):
        return self.img_id

    @property
    def im_path(self):
        return self._complete_path('{}/{}')

    @property
    def objects(self):
        if self._objects is None:
            self._objects = [KITTIObj.deserialize(line.strip()) for line in self.gt]
        return self._objects


class KITTISequence(Dataset):
    """
    Class holding datasets containing KITTIDatum objects.
    """

    def __init__(self, root_path, seq_id):
        """

        Args:
            root_path:      path to training folder
            seq_id:         str of len 4 (eg: 0001)
        """
        _ids = sorted(os.listdir(os.path.join(root_path, 'image_02', seq_id)))
        self._all_img_ids = [x for x in _ids if x.endswith((".jpg", ".png")) and not x.startswith(".")]
        self._gt = defaultdict(list)
        self.seq_id = seq_id
        with open("{}/label_02/{}.txt".format(root_path, seq_id)) as fo:
            lines = fo.readlines()
            for line in lines:
                _temp = line.strip().split(' ')
                if _temp[2].lower() == 'dontcare':
                    continue
                idx = int(_temp[0])
                self._gt[idx].append(line)

        super(KITTISequence, self).__init__(root_path, KITTIDatum)

    @property
    def all_img_ids(self):
        return self._all_img_ids

    def create_datum(self, img_id: str) -> Datum:
        image_dir = os.path.join(self.root_path, "image_02", self.seq_id)
        return self._datum_class(image_dir, img_id, self._gt[int(img_id[:-4])])


class KITTIDataset:
    """
    Class for holding KITTI sequences.
    """

    def __init__(self, root_path):
        """

        Args:
            root_path:      path to folder containing sequence sub-folders
        """
        self.root_path = root_path
        self.seq_folders = sorted([x for x in os.listdir(os.path.join(self.root_path,
                                                                      "image_02")) if not x.startswith(".")])

    def get_sequence(self, seq_id):
        """

        Args:
            seq_id:     str (seq folder name)

        Returns:
            KITTISequence object instance
        """
        seq = KITTISequence(root_path=self.root_path, seq_id=seq_id)
        return seq

    @property
    def sequences(self):
        return [self.get_sequence(seq_id) for seq_id in self.seq_folders]
