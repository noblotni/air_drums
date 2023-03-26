"""UI elements."""
from threading import Thread
import numpy as np
import cv2 as cv
import simpleaudio as sa
from air_drums.hand_tracking import CentroidTracker

MIN_DIFF = 20


def draw_drums(frame):
    """Draw areas where the user's hands can produce sounds."""
    # Draw rectangles
    for i in range(3):
        cv.rectangle(
            frame,
            (0, i * frame.shape[0] // 3),
            (frame.shape[1] // 3, (i + 1) * frame.shape[0] // 3),
            color=(255, 0, 0),
            thickness=2,
        )
        cv.rectangle(
            frame,
            (2 * frame.shape[1] // 3, i * frame.shape[0] // 3),
            (frame.shape[1], (i + 1) * frame.shape[0] // 3),
            color=(255, 0, 0),
            thickness=2,
        )
    return frame


def play_drum(drum: sa.WaveObject):
    drum_play = drum.play()
    drum_play.wait_done()


def draw_points(tracker: CentroidTracker, frame: np.ndarray, drum: sa.WaveObject):
    for _, obj in tracker.objects.items():
        if obj.diff_dist_topleft_corner > MIN_DIFF:
            cv.circle(
                frame, center=obj.center, radius=0, thickness=20, color=(0, 0, 255)
            )
            thread = Thread(target=play_drum, args=(drum,))
            thread.start()
    return frame
