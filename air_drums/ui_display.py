"""UI elements."""
import cv2 as cv


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
