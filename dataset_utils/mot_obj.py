from dataset_utils.label_obj import BaseObject


class MOTObj(BaseObject):

    @classmethod
    def deserialize(cls, line):
        """
        Alternate constructor to initialize from a string.
        Args:
            line:   string containing <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Returns:
            RotationObject instance
        """
        line = line.strip().split(',')
        return cls(x_min=float(line[2]),
                   y_min=float(line[3]),
                   x_max=float(line[4]) + float(line[2]),
                   y_max=float(line[5]) + float(line[3]),
                   category="unknown",
                   frame=int(line[0]),
                   track=int(line[1]),
                   confidence=float(line[6]),
                   x_3d=float(line[7]),
                   y_3d=float(line[8]),
                   z_3d=float(line[9])
                   )

    @classmethod
    def deserialize_gt(cls, line):
        """
        Alternate constructor to initialize from a string.
        Args:
            line:   string containing <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, ...

        Returns:
            RotationObject instance
        """
        line = line.strip().split(',')
        return cls(x_min=float(line[2]),
                   y_min=float(line[3]),
                   x_max=float(line[4]) + float(line[2]),
                   y_max=float(line[5]) + float(line[3]),
                   category="unknown",
                   frame=int(line[0]),
                   track=int(line[1])
                   )
