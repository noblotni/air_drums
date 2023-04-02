"""Entry point of the package"""
import cv2 as cv
import simpleaudio as sa
from air_drums.hand_detection import new_detect_hands
from air_drums.hand_tracking import CentroidTracker
from air_drums.ui_display import draw_drums, hit_event, display_drums_text


def main():
    """Run the air drums algorithm."""
    tracker = CentroidTracker()
    bg_subtractor = cv.createBackgroundSubtractorMOG2()
    video = cv.VideoCapture(0)
    while True:
        _, frame = video.read()
        frame, obj_rects = new_detect_hands(frame, bg_subtractor)
        frame = draw_drums(frame)
        frame = display_drums_text(frame)
        tracker.update(obj_rects)
        if len(tracker.objects) > 0:
            frame = hit_event(tracker=tracker, frame=frame)
        frame = cv.resize(frame, (2 * frame.shape[1], 2 * frame.shape[0]))
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
