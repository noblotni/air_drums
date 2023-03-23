"""Entry point of the package"""
import cv2 as cv
from air_drums.hand_detection import detect_hands
from air_drums.ui_display import draw_drums


def main():
    """Run the air drums algorithm."""
    video = cv.VideoCapture(0)
    while True:
        _, frame = video.read()
        frame = detect_hands(frame)
        frame = draw_drums(frame)
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
