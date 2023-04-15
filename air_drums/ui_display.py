"""UI elements."""
import numpy as np
import cv2 as cv
from air_drums.hand_tracking import CentroidTracker
import air_drums.sound_player as sp
from air_drums.sound_player import play_drum_sound

MIN_DIFF = 20


def create_drums_rectangles_coords(frame: np.ndarray):
    drums_rects_coords = {}
    drums_rects_coords["Kick"] = np.array(
        [0, 0, frame.shape[1] // 3, frame.shape[0] // 3]
    )
    drums_rects_coords["Bass"] = np.array(
        [
            0,
            frame.shape[0] // 3,
            frame.shape[1] // 3,
            frame.shape[0] // 3,
        ]
    )
    drums_rects_coords["Clap"] = np.array(
        [
            0,
            2 * frame.shape[0] // 3,
            frame.shape[1] // 3,
            frame.shape[0] // 3,
        ]
    )
    drums_rects_coords["Cymbal"] = np.array(
        [
            2 * frame.shape[1] // 3,
            0,
            frame.shape[1] // 3,
            frame.shape[0] // 3,
        ]
    )
    drums_rects_coords["Tom"] = np.array(
        [
            2 * frame.shape[1] // 3,
            frame.shape[0] // 3,
            frame.shape[1] // 3,
            frame.shape[0] // 3,
        ]
    )
    drums_rects_coords["Snare"] = np.array(
        [
            2 * frame.shape[1] // 3,
            2 * frame.shape[0] // 3,
            frame.shape[1] // 3,
            frame.shape[0] // 3,
        ]
    )
    return drums_rects_coords


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


def display_drums_text(frame: np.ndarray):
    """Display the name of the drums."""
    frame = cv.putText(
        frame,
        text="Kick",
        org=(frame.shape[1] // 7, frame.shape[0] // 6),
        fontScale=2,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        thickness=2,
    )
    frame = cv.putText(
        frame,
        text="Bass",
        org=(frame.shape[1] // 7, frame.shape[0] // 2),
        fontScale=2,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        thickness=2,
    )
    frame = cv.putText(
        frame,
        text="Clap",
        org=(frame.shape[1] // 7, 5 * frame.shape[0] // 6),
        fontScale=2,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        thickness=2,
    )
    frame = cv.putText(
        frame,
        text="Cymbal",
        org=(5 * frame.shape[1] // 7, frame.shape[0] // 6),
        fontScale=2,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        thickness=2,
    )
    frame = cv.putText(
        frame,
        text="Tom",
        org=(5 * frame.shape[1] // 7, frame.shape[0] // 2),
        fontScale=2,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        thickness=2,
    )
    frame = cv.putText(
        frame,
        text="Snare",
        org=(5 * frame.shape[1] // 7, 5 * frame.shape[0] // 6),
        fontScale=2,
        fontFace=cv.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        thickness=2,
    )
    return frame


def is_in_rectangle(point: np.ndarray, rect: np.ndarray):
    return (rect[0] <= point[0] <= rect[0] + rect[2]) and (
        rect[1] <= point[1] <= rect[1] + rect[3]
    )


def find_hit_drum(frame: np.ndarray, center: np.ndarray):
    drums_rects_coords = create_drums_rectangles_coords(frame)
    for drum_key, rect in drums_rects_coords.items():
        hit = is_in_rectangle(center, rect)
        if hit and drum_key == "Kick":
            return sp.BOOM_KICK
        elif hit and drum_key == "Bass":
            return sp.BASS_DRUM
        elif hit and drum_key == "Cymbal":
            return sp.CRASH_CYMBAL
        elif hit and drum_key == "Clap":
            return sp.CLAP
        elif hit and drum_key == "Tom":
            return sp.ELECTRONIC_TOM
        elif hit and drum_key == "Snare":
            return sp.SNARE_DRUM


def hit_event(tracker: CentroidTracker, frame: np.ndarray):
    for _, obj in tracker.objects.items():
        if obj.diff_dist > MIN_DIFF:
            cv.circle(
                frame, center=obj.center, radius=0, thickness=20, color=(0, 0, 255)
            )
            drum = find_hit_drum(frame, obj.center)
            play_drum_sound(drum)
    return frame
