from matplotlib import pyplot as plt
from vis_utils.vis_utils import draw_bounding_box_on_image
from PIL import Image


def vis_datum_mask(dataset):
    datum = dataset.random_datum()
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(datum.image)
    fig.add_subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(datum.image)
    plt.imshow(datum.mask, alpha=50)


def datum_with_labels(datum, objects=None):
    """
    Draws bounding boxes on image.
    Args:
        datum:      Datum Object
        objects:    List of object instances

    Returns:
        PIL Image instance
    """
    if objects is None:
        boxes = [[obj.y_min, obj.x_min, obj.y_max, obj.x_max] for obj in datum.objects]
    else:
        boxes = [[obj.y_min, obj.x_min, obj.y_max, obj.x_max] for obj in objects]

    img = datum.image
    for box in boxes:
        draw_bounding_box_on_image(img, box[0], box[1], box[2], box[3], use_normalized_coordinates=False)

    return img


def datum_with_track(datum, track=None):
    """
    Draws bounding boxes on image.
    Args:
        datum:      Datum Object
        track:    List of object instances

    Returns:
        PIL Image instance
    """

    boxes = [[obj.y_min, obj.x_min, obj.y_max, obj.x_max] for obj in datum.ground_truth if obj.track == track]

    img = datum.image
    for box in boxes:
        draw_bounding_box_on_image(img, box[0], box[1], box[2], box[3], use_normalized_coordinates=False)

    return img


def image_with_box(path, box, image=None, color='red', thickness=2):
    """

    Args:
        path:       str (path to image)
        box:        [x, y, w, h]
                    x,y are normalized centre / w,h also normalized
        image:      give image to ignore path
        color:      box color
        thickness:  box thinkness

    Returns:

    """
    if image is None:
        image = Image.open(path)
    x, y, w, h = box[0], box[1], box[2], box[3]
    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2
    draw_bounding_box_on_image(image=image, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, color=color,
                               thickness=thickness)

    return image


class ImageBoxes:

    def __init__(self, path, plt_axis_off=False):
        if plt_axis_off:
            _, self.ax = plt.subplots(1)
            plt.axis("off")
        self.im = Image.open(path)

    def add_box(self, box, color='red', thickness=2, flip_order=False):
        if flip_order:
            cx, cy, h, w = box
            box = cx, cy, w, h
        self.im = image_with_box(path='', box=box, image=self.im, color=color, thickness=thickness)

    def get_final(self):
        return self.im.convert("RGB")

    def add_from_track(self, x, y, y_pred):
        """
        Args:
            x:          x of shape (time_steps, features)
            y:          y of shape (features,)
            y_pred:     y_pred of shape (features,)
        """
        for i in x[:, :4]:
            self.add_box(list(i[:4]), color='blue', flip_order=True)
        self.add_box(list(y[:4]), color='red', thickness=4, flip_order=True)
        self.add_box(list(y_pred[:4]), color='orange', thickness=4, flip_order=True)

    def show_plt(self, arr):
        """
        Plots image in current axis
        Args:
            arr:        image as np.array
        """
        self.ax.imshow(arr)
