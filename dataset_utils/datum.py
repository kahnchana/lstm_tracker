from __future__ import division

from typing import List

import numpy as np
from PIL import Image

from dataset_utils.label_obj import BaseObject


class Datum(object):
    """
    This is an abstraction of one observation (an image with groundtruth labels and other data) in a dataset like KITTI.
    """

    def __init__(self, root_path, img_id, fixed_im_size=None):

        self.root_path = root_path
        self.img_id = img_id
        self._fixed_im_size = fixed_im_size

        # Placeholders for cached data
        self._rgb = None
        self._objects = None
        self._camdata = None
        self._objectmask = None
        self._classmask = None

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------- Abstract Methods --------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    @property
    def title(self):
        """
        Return the title of the datum, which usually corresponds to the file title. Usually this is derived from the
        self.img_id property.

        e.g. For KITTI, title whill be '000008' for img_id = 8

        Returns:
            Title of the datum
        """
        raise NotImplementedError('To be implemented by a subclass')

    @property
    def im_path(self):
        """
        Path to the RGB image

        Returns:
            Path to the RGB image
        """
        raise NotImplementedError('To be implemented by a subclass')

    @property
    def objects(self) -> List[BaseObject]:
        """
        Returns groundtruth objects associated with this datum.

        Returns:
            List of groundtruth objects in the image
        """
        raise NotImplementedError('To be implemented by a subclass')

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------- Concrete Methods --------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    @property
    def rgb(self):
        """
        Return the RGB values of the image as a numpy array. The array will be internally cached and therefore should
        not be modified directly.

        Returns:
            np.array RGB values of the left color camera. Has shape [height, width, channels=3]
        """
        if self._rgb is None:
            self._rgb = np.array(Image.open(self.im_path))[:, :, :3]

            if self._fixed_im_size is not None:
                height, width = self._fixed_im_size
                self._rgb = self._rgb[:height, :width, :]

        return self._rgb

    @property
    def image(self):
        """
        Returns the left color camera image as a PIL image. Return value is not cached and a fresh copy of the PIL
        image is made everytime this method is called. Therefore output of this method can be modified (using ImageDraw
        methods, for example)

        Returns:
            Image PIL image of the left color camera
        """
        return Image.open(self.im_path)

    def _complete_path(self, fmt):
        """
        Completes the path by filling in the root_path and title of the datum
        E.g.complete_path('{}/image_2/{}.png') -> '/home/sadeep/dataset_root/image_2/000005.png'

        Args:
            fmt: Format of the path to be completed.

        Returns:
            Completed path
        """
        return fmt.format(self.root_path, self.title)

    def show_im(self):
        Image.fromarray(self.rgb).show()

    @staticmethod
    def load_image(path):
        """
        Use to load segmentation masks
        Args:
            path:   path to image as str

        Returns:
            PIL Image object
        """
        return Image.open(path)
