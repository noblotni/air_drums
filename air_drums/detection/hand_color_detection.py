import numpy as np
import cv2 as cv

GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA_X = 1
HSV_SKIN_LOW_THRESHOLD = np.array([2, 50, 50])
HSV_SKIN_HIGH_THRESHOLD = np.array([15, 255, 255])
MIN_CNT_AREA = 5000


def detect_skin(frame: np.ndarray):
    """Detect the skin of the user."""
    # Convert to hsv space
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    skin_mask = cv.inRange(hsv_frame, HSV_SKIN_LOW_THRESHOLD, HSV_SKIN_HIGH_THRESHOLD)
    return skin_mask


def denoise_morphology(frame: np.ndarray):
    """Apply morphological operations and filtering to denoise the frame."""
    # Blur image
    frame = cv.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X)
    kernel_square = np.ones((11, 11))
    kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    frame = cv.dilate(frame, kernel=kernel_ellipse)
    frame = cv.erode(frame, kernel=kernel_square, iterations=2)
    frame = cv.dilate(frame, kernel=kernel_ellipse)
    frame = cv.medianBlur(frame, 5)
    kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    frame = cv.dilate(frame, kernel=kernel_ellipse)
    frame = cv.medianBlur(frame, 5)
    kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    frame = cv.dilate(frame, kernel=kernel_ellipse)
    frame = cv.medianBlur(frame, 5)
    return frame


def threshold_frame(frame: np.ndarray):
    _, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Mask the central area where the user's head is not to detect it
    frame[:, frame.shape[1] // 3 : 2 * frame.shape[1] // 3] = 0
    return frame


def find_hands_contours(frame: np.ndarray, skin_mask: np.ndarray):
    """Find the contours of the user's hands."""
    contours, _ = cv.findContours(skin_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    areas = []
    # List to store the bounds of the objects found
    obj_rects = []
    for contour in contours:
        cnt_area = cv.contourArea(contour)
        if cnt_area > MIN_CNT_AREA:
            areas.append(cnt_area)
            approx = cv.approxPolyDP(contour, 0.001 * cv.arcLength(contour, True), True)
            cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
            center = 1 / 4 * (box[0, :] + box[1, :] + box[2, :] + box[3, :])
            center = center.astype(int)
            cv.circle(frame, center=center, radius=0, color=(255, 0, 0), thickness=10)
            obj_rects.append(box)

    return frame, obj_rects


def detect_color_hands(frame: np.ndarray):
    """Detect the hands on the frame."""
    skin_mask = detect_skin(frame)
    skin_mask = denoise_morphology(skin_mask)
    skin_mask = threshold_frame(skin_mask)
    # Flip the mask and the frame horizontally
    skin_mask = cv.flip(skin_mask, 1)
    frame = cv.flip(frame, 1)
    frame, obj_rects = find_hands_contours(frame=frame, skin_mask=skin_mask)
    return frame, obj_rects
