import numpy as np
import cv2

def blend(frame, warped, corners):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(int), 255)
    
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask_inv)

    warped_fg = cv2.bitwise_and(warped, warped, mask)
    return cv2.add(frame_bg, warped_fg)

def insert_image(frame, insert_img, corners) -> np.ndarray:
    h_image, w_image = insert_img.shape[:2]
    src_points = np.float32([
        [0,   0  ],   # top-left
        [w_image-1, 0  ],   # top-right
        [w_image-1, h_image-1],   # bottom-right
        [0,   h_image-1]    # bottom-left
    ])

    transform = cv2.getPerspectiveTransform(src_points, corners)
    h_frame, w_frame = frame.shape[:2]
    warped = cv2.warpPerspective(insert_img, transform, (w_frame, h_frame))

    return blend(frame, warped, corners)