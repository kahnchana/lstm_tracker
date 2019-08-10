from dataset_utils.label_obj import BaseObject


class KITTIObj(BaseObject):

    def __init__(self, x_min, y_min, x_max, y_max, category, truncated, occluded, frame, track, x_3d, y_3d, z_3d, width,
                 height, length, yaw, observed_angle, pose=None, difficult=None, confidence=None, score=None):
        """
        Create Base Object for KITTI Tracking.

        Args:
            x_min:
            y_min:
            x_max:
            y_max:
            category:
            truncated:
            pose:
            difficult:
            frame:
            track:
            confidence:
            x_3d:
            y_3d:
            z_3d:
            width:
            height:
            length:
            yaw:
            observed_angle:
            score:
        """
        self.width = width
        self.height = height
        self.length = length
        self.yaw = yaw
        self.observed_angle = observed_angle
        self.occluded = occluded
        self.score = score
        self.ignored = False
        self.valid = False
        self.tracker = -1

        super(KITTIObj, self).__init__(
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, category=category, truncated=truncated, pose=pose,
            difficult=difficult, frame=frame, track=track, confidence=confidence, x_3d=x_3d, y_3d=y_3d, z_3d=z_3d)

    def serialize(self):
        attrs = vars(self)
        return ", ".join(["{}: {}".format(n, v) for n, v in attrs.items()])

    @classmethod
    def deserialize(cls, line):
        """
        Alternate constructor to initialize from a string.
        Args:
            line:       KITTI tracker benchmark data format
                        (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)

        Returns:
            RotationObject instance
        """
        fields = line.strip().split(" ")
        frame = int(float(fields[0]))  # frame
        track_id = int(float(fields[1]))  # id
        obj_type = fields[2].lower()  # object type [car, pedestrian, cyclist, ...]
        truncation = int(float(fields[3]))  # truncation [-1,0,1,2]
        occlusion = int(float(fields[4]))  # occlusion  [-1,0,1,2]
        obs_angle = float(fields[5])  # observation angle [rad]
        x1 = float(fields[6])  # left   [px]
        y1 = float(fields[7])  # top    [px]
        x2 = float(fields[8])  # right  [px]
        y2 = float(fields[9])  # bottom [px]
        h = float(fields[10])  # height [m]
        w = float(fields[11])  # width  [m]
        l = float(fields[12])  # length [m]
        X = float(fields[13])  # X [m]
        Y = float(fields[14])  # Y [m]
        Z = float(fields[15])  # Z [m]
        yaw = float(fields[16])  # yaw angle [rad]

        return cls(x_min=x1,
                   y_min=y1,
                   x_max=x2,
                   y_max=y2,
                   height=h,
                   width=w,
                   length=l,
                   x_3d=X,
                   y_3d=Y,
                   z_3d=Z,
                   yaw=yaw,
                   frame=frame,
                   track=track_id,
                   category=obj_type,
                   truncated=truncation,
                   occluded=occlusion,
                   observed_angle=obs_angle)
