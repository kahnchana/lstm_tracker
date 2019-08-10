class BaseObject(object):
    """
    Defines a base object class for holding individual object data
    """

    def __init__(self, x_min, y_min, x_max, y_max, category, truncated=None, pose=None, difficult=None, frame=None,
                 track=None, confidence=None, x_3d=None, y_3d=None, z_3d=None):
        """
        Initiates object.
        Args:
            x_min:          top left x coordinate
            y_min:          top left y coordinate
            x_max:          bottom right x coordinate
            y_max:          bottom right y coordinate
            category:       class as str
            truncated:      int
            pose:           int
            difficult:      int
            frame:          int
            track:          int
            confidence:     float
            x_3d:           float
            y_3d:           float
            z_3d:           float
        """
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.category = category
        self.truncated = truncated
        self.pose = pose
        self.difficult = difficult
        self.frame = frame
        self.track = track
        self.confidence = confidence
        self.x_3d = x_3d
        self.y_3d = y_3d
        self.z_3d = z_3d

    def __repr__(self):
        return self.serialize()

    def __eq__(self, other):
        return self.serialize() == other.serialize()

    def serialize(self):
        return '{:.2f} {:.2f} {:.2f} {:.2f} {} {} {} {} {} {} {} {} {} {}'.format(
            self.x_min, self.y_min, self.x_max, self.y_max, self.category, self.truncated, self.pose, self.difficult,
            self.frame, self.track, self.confidence, self.x_3d, self.y_3d, self.z_3d)

    @classmethod
    def deserialize(cls, line):
        """
        Alternate constructor to initialize from a string.
        Args:
            line:   string containing "x1 y1 x2 y2 category"

        Returns:
            RotationObject instance
        """
        line = line.strip().split(' ')
        return cls(x_min=float(line[0]),
                   y_min=float(line[1]),
                   x_max=float(line[2]),
                   y_max=float(line[3]),
                   category=line[4])
