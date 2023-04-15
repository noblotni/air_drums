"""Entry point of the package."""
import argparse
import cv2 as cv
from air_drums.detection.hand_color_detection import detect_color_hands
from air_drums.detection.hand_motion_detection import detect_motion_hands
from air_drums.hand_tracking import CentroidTracker
from air_drums.ui_display import draw_drums, hit_event, display_drums_text


def main(args):
    """Run the air drums algorithm."""
    tracker = CentroidTracker()
    if args.detector == "motion":
        bg_subtractor = cv.createBackgroundSubtractorMOG2()
    video = cv.VideoCapture(0)
    while True:
        _, frame = video.read()
        if args.detector == "motion":
            frame, obj_rects = detect_motion_hands(frame, bg_subtractor)
        elif args.detector == "color":
            frame, obj_rects = detect_color_hands(frame)
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
    parser = argparse.ArgumentParser("AirDrums algorithm.")
    parser.add_argument(
        "--detector",
        choices=["color", "motion"],
        default="color",
        help="Type of detector. (default: color)",
    )
    args = parser.parse_args()
    main(args)
