import numpy as np
import cv2
import argparse
from sklearn.cluster import DBSCAN

def polygon_angles_deg(pts):
    pts = np.asarray(pts, dtype=np.float32)
    angles = []

    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p = pts[i]
        p_next = pts[(i + 1) % 4]

        v1 = p_prev - p
        v2 = p_next - p

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
            continue

        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        angles.append(float(angle))

    return np.array(angles, dtype=np.float32)


def expand_polygon_from_center(pts, scale):
    pts = np.asarray(pts, dtype=np.float32)
    center = pts.mean(axis=0, keepdims=True)
    return (center + (pts - center) * scale).astype(np.float32)

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis = 1).reshape(-1)
    top_left = pts[np.argmin(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]
    bottom_right = pts[np.argmax(s)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
def find_display_quad_contour(image, min_area=20000, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Небольшой прикол, чтобы склеить разрывы на границах
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1e18
    best_contour = None
    best_box = None

    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Берем только четырехугольники
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        pts = order_points(pts)

        # Размеры сторон
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])

        width = 0.5 * (width_top + width_bottom)
        height = 0.5 * (height_left + height_right)

        if width < 60 or height < 60:
            continue

        aspect_ratio = max(width, height) / max(1.0, min(width, height))
        if not (1.0 <= aspect_ratio <= 3.5):
            continue

        # Центр кандидата
        center = pts.mean(axis=0)
        cx, cy = center

        # Слишком у края кадра не берем
        if not (0.10 * w <= cx <= 0.90 * w and 0.10 * h <= cy <= 0.95 * h):
            continue

        # Насколько углы близки к 90°
        angles = polygon_angles_deg(pts)
        angle_error = np.mean(np.abs(angles - 90.0))

        # Слишком кривые четырехугольники отбрасываем
        if angle_error > 25.0:
            continue

        # Маска внутренности
        inner_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(inner_mask, pts.astype(np.int32), 255)

        # Внутренность чуть ужмем, чтобы не цеплять рамку
        inner_shrunk = expand_polygon_from_center(pts, 0.92)
        inner_mask_shrunk = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(inner_mask_shrunk, inner_shrunk.astype(np.int32), 255)

        # Внешнее кольцо вокруг кандидата
        outer_expanded = expand_polygon_from_center(pts, 1.08)
        outer_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(outer_mask, outer_expanded.astype(np.int32), 255)

        ring_mask = cv2.subtract(outer_mask, inner_mask)

        # Средняя яркость внутри и снаружи
        mean_inside = cv2.mean(gray, mask=inner_mask_shrunk)[0]
        mean_outside = cv2.mean(gray, mask=ring_mask)[0] if np.count_nonzero(ring_mask) > 0 else mean_inside

        # Экран обычно темнее, чем область сразу вокруг него
        contrast_dark = mean_outside - mean_inside

        border_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.polylines(border_mask, [pts.astype(np.int32)], True, 255, 3)
        border_edge_strength = cv2.mean(edges, mask=border_mask)[0]
        
        center_dist = np.linalg.norm(center - img_center) / max(1.0, np.linalg.norm([w, h]))

        # Насколько контур близок к своей площади (не рваный ли)
        box_area = cv2.contourArea(pts.astype(np.int32))
        fill_ratio = area / max(box_area, 1.0)

        score = (
            1.5 * area
            + 250.0 * border_edge_strength
            + 180.0 * contrast_dark
            + 20000.0 * min(fill_ratio, 1.0)
            - 2500.0 * angle_error
            - 30000.0 * center_dist
        )

        if score > best_score:
            best_score = score
            best_contour = cnt
            best_box = pts

    if best_box is None:
        return None, None

    if debug:
        dbg = image.copy()
        cv2.polylines(dbg, [best_box.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite("debug_quad_detector.jpg", dbg)

    return best_contour, best_box

def detect_sift_points(gray, roi_rect, max_points = 300):
    x0, y0, x1, y1 = roi_rect
    roi = gray[y0:y1, x0:x1]

    sift = cv2.SIFT_create(nfeatures=max_points)
    keypoints, descriptors = sift.detectAndCompute(roi, None)

    if not keypoints:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    pts = []
    responses = []
    for kp in keypoints:
        x, y = kp.pt
        pts.append([x + x0, y + y0])
        responses.append(kp.response)

    pts = np.array(pts, dtype=np.float32)
    responses = np.array(responses, dtype=np.float32)

    return pts, responses

def cluster_corner_points_dbscan(points, responses, rough_corner, radius = 80.0, eps = 18.0, min_samples = 2):
    if len(points) == 0:
        return rough_corner.copy()
    dists = np.linalg.norm(points - rough_corner, axis=1)
    mask = dists <= radius

    local_points = points[mask]
    local_responses = responses[mask]

    if len(local_points) == 0:
        return rough_corner.copy()

    if len(local_points) < min_samples:
        idx = np.argmin(np.linalg.norm(local_points - rough_corner, axis=1))
        return local_points[idx].astype(np.float32)
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(local_points)
    unique_labels = [lb for lb in np.unique(labels) if lb != -1]

    if not unique_labels:
        idx = np.argmin(np.linalg.norm(local_points - rough_corner, axis=1))
        return local_points[idx].astype(np.float32)

    best_label = None
    best_score = None

    for lb in unique_labels:
        cluster_pts = local_points[labels == lb]
        cluster_resp = local_responses[labels == lb]

        centroid = cluster_pts.mean(axis=0)
        centroid_dist = np.linalg.norm(centroid - rough_corner)

        score = (
            -len(cluster_pts),          # чем больше кластер, тем лучше
            centroid_dist,              # чем ближе центр, тем лучше
            -float(cluster_resp.mean()) # чем сильнее response, тем лучше
        )
        if best_score is None or score < best_score:
            best_score = score
            best_label = lb

    cluster_pts = local_points[labels == best_label]
    cluster_resp = local_responses[labels == best_label]

    centroid = cluster_pts.mean(axis=0)
    centroid_dists = np.linalg.norm(cluster_pts - centroid, axis=1)
    cd = centroid_dists / max(1e-6, centroid_dists.max()) if len(cluster_pts) > 1 else np.array([0.0])
    cr = cluster_resp / max(1e-6, cluster_resp.max())

    point_score = cd - 0.5 * cr
    best_idx = np.argmin(point_score)
        
    return cluster_pts[best_idx].astype(np.float32)

def accept_refined_corner(rough_corner, refined_corner, max_shift=18.0):
    shift = np.linalg.norm(refined_corner - rough_corner)
    if shift > max_shift:
        return rough_corner.copy()
    return refined_corner

def point_to_segment_distance(p, a, b):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    ab = b - a
    ab_len2 = float(np.dot(ab, ab))

    if ab_len2 < 1e-6:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / ab_len2)
    t = max(0.0, min(1.0, t))

    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def filter_points_for_corner(points, responses, rough_box, corner_idx, max_dist_to_edge=12.0):
    if len(points) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    tl, tr, br, bl = rough_box
    if corner_idx == 0:          # top-left
        seg1 = (tl, tr)          # верхняя
        seg2 = (tl, bl)          # левая
    elif corner_idx == 1:        # top-right
        seg1 = (tl, tr)          # верхняя
        seg2 = (tr, br)          # правая
    elif corner_idx == 2:        # bottom-right
        seg1 = (tr, br)          # правая
        seg2 = (bl, br)          # нижняя
    elif corner_idx == 3:        # bottom-left
        seg1 = (tl, bl)          # левая
        seg2 = (bl, br)          # нижняя
    else:
        raise ValueError("corner_idx must be 0, 1, 2, or 3")

    keep_idx = []

    for i, p in enumerate(points):
        d1 = point_to_segment_distance(p, seg1[0], seg1[1])
        d2 = point_to_segment_distance(p, seg2[0], seg2[1])

        if d1 <= max_dist_to_edge and d2 <= max_dist_to_edge:
            keep_idx.append(i)

    if len(keep_idx) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    keep_idx = np.array(keep_idx, dtype=np.int32)
    return points[keep_idx], responses[keep_idx]

def find_display(image: np.ndarray, debug: bool = True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contour, rough_box = find_display_quad_contour(image, debug=debug)
    if rough_box is None:
        return None

    rough_box = order_points(rough_box)

    xs = rough_box[:, 0]
    ys = rough_box[:, 1]
    pad = 30

    x0 = max(0, int(np.min(xs) - pad))
    y0 = max(0, int(np.min(ys) - pad))
    x1 = min(image.shape[1], int(np.max(xs) + pad))
    y1 = min(image.shape[0], int(np.max(ys) + pad))

    sift_points, sift_responses = detect_sift_points(gray, (x0, y0, x1, y1), max_points=300)

    final_corners = []
    for i, rc in enumerate(rough_box):
        corner_points, corner_responses = filter_points_for_corner(
            sift_points,
            sift_responses,
            rough_box,
            corner_idx=i,
            max_dist_to_edge=12.0,
        )
    
        pt = cluster_corner_points_dbscan(
            corner_points,
            corner_responses,
            rc,
            radius=60.0,
            eps=12.0,
            min_samples=3,
        )
    
        pt = accept_refined_corner(rc, pt, max_shift=18.0)
        final_corners.append(pt)

    final_corners = np.array(final_corners, dtype=np.float32)
    final_corners = order_points(final_corners)

    if debug:
        dbg = image.copy()

        cv2.polylines(dbg, [rough_box.astype(np.int32)], True, (0, 255, 0), 2)

        # Все SIFT точки в ROI
        for p in sift_points:
            cv2.circle(dbg, tuple(p.astype(int)), 3, (255, 0, 0), -1)

        for i, p in enumerate(final_corners):
            cv2.circle(dbg, tuple(p.astype(int)), 8, (0, 0, 255), -1)
            cv2.putText(
                dbg,
                str(i),
                (int(p[0]) + 8, int(p[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite("debug_sift_dbscan_result.jpg", dbg)

    return final_corners


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frame", default=None)
    args = parser.parse_args()
    if args.frame is None:
        raise "I need frame! [--frame]"
    img = cv2.imread(args.frame)
    corners = find_display(img, debug=True)

    if corners is None:
        print("Экран не найден")
    else:
        print("Corners (TL, TR, BR, BL):")
        for p in corners.astype(int):
            print(tuple(p))

