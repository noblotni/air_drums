"""Track the hands."""
import numpy as np
import uuid

SAME_OBJECT_DIST = 30
MAX_OBJ_NOT_FOUND = 3


class ObjectDetected:
    def __init__(self, center: np.ndarray, rectangle: np.ndarray):
        self.rectangle = rectangle
        self.center = center
        # Boolean which indicates if an object is found or not once created
        self.object_found = True
        # Counter to discard the object if not found three times in a row
        self.object_not_found_count = 0
        self.current_dist_topleft_corner = np.linalg.norm(rectangle[0, :] - self.center)
        # Quantify zoom
        self.diff_dist_topleft_corner = 0


class CentroidTracker:
    def __init__(self) -> None:
        self.objects: dict[ObjectDetected] = {}

    def reset_objects(self):
        """Set object_found to False for all the objects."""
        for _, obj in self.objects.items():
            obj.object_found = False

    def remove_objects(self):
        """Delete object if not found for a too long time."""
        to_remove = []
        for id_obj, obj in self.objects.items():
            if not obj.object_found:
                obj.object_not_found_count += 1
            if obj.object_not_found_count > MAX_OBJ_NOT_FOUND:
                to_remove.append(id_obj)

        for id_obj in to_remove:
            del self.objects[id_obj]

    def update(self, obj_rects: list):
        self.reset_objects()
        for rect in obj_rects:
            same_object = False
            new_center = 1 / 4 * (rect[0, :] + rect[1, :] + rect[2, :] + rect[3, :])
            new_center = new_center.astype(int)
            for _, obj in self.objects.items():
                old_center = obj.center
                distance_centers = np.linalg.norm(old_center - new_center)
                if distance_centers < SAME_OBJECT_DIST:
                    same_object = True
                    obj.center = new_center
                    obj.rectangle = rect
                    dist = np.linalg.norm(rect[0, :] - new_center)
                    obj.diff_dist_topleft_corner = (
                        dist - obj.current_dist_topleft_corner
                    )
                    obj.current_dist_topleft_corner = dist

            if not same_object:
                new_id = uuid.uuid1().int
                self.objects[new_id] = ObjectDetected(rectangle=rect, center=new_center)
        self.remove_objects()
