import numpy as np
import cv2 as cv

MIN_CNT_AREA = 5000


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


def detect_motion_hands(frame: np.ndarray, bg_subtractor):
    skin_mask = bg_subtractor.apply(frame)
    skin_mask = cv.flip(skin_mask, 1)
    # Mask the central area where the user's head is not to detect it
    skin_mask[:, frame.shape[1] // 3 : 2 * frame.shape[1] // 3] = 0
    frame = cv.flip(frame, 1)
    frame, obj_rects = find_hands_contours(frame, skin_mask)
    return frame, obj_rects
