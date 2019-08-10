import numpy as np

from dataset_utils.datum import Datum


class Dataset(object):
    def __init__(self, root_path, datum_class=None):
        # Public properties
        self.root_path = root_path

        # Private ones
        self._datum_class = datum_class
        self._size = None

    @property
    def all_img_ids(self):
        raise NotImplementedError('To be implemented by subclass')

    @property
    def size(self):
        if self._size is None:
            self._size = len(self.all_img_ids)
        return self._size

    def create_datum(self, img_id: str) -> Datum:
        return self._datum_class(self.root_path, img_id)

    def random_datum(self):
        """
        Returns a random Datum from this dataset, useful for debugging/testing.

        Returns:
            Datum A random datum from this dataset
        """
        rand_img_id = np.random.choice(self.all_img_ids)
        return self.create_datum(rand_img_id)

    def __len__(self):
        return self.size

    def __str__(self):
        return "Dataset(root_path=%s)" % self.root_path

    def datums(self, shuffle=False):
        ids = self.all_img_ids
        if shuffle:
            np.random.shuffle(ids)
        for img_id in ids:
            yield self.create_datum(img_id)
