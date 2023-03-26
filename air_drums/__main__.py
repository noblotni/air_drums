"""Entry point of the package"""
import cv2 as cv
import simpleaudio as sa
from air_drums.hand_detection import detect_hands
from air_drums.hand_tracking import CentroidTracker
from air_drums.ui_display import draw_drums, draw_points


def main():
    """Run the air drums algorithm."""
    tracker = CentroidTracker()
    drum = sa.WaveObject.from_wave_file("./assets/audio/Clap-1.wav")
    video = cv.VideoCapture(0)
    while True:
        _, frame = video.read()
        frame, obj_rects = detect_hands(frame)
        frame = draw_drums(frame)
        tracker.update(obj_rects)
        if len(tracker.objects) > 0:
            frame = draw_points(tracker=tracker, frame=frame, drum=drum)
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
