import display_detection
import image_warping
import cv2
import argparse
import textwrap
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Insert image to display
            -------------------
            Inserts image to display found on frame from a camera, video file, or single image.
            If multiple sources are given, priority is: camera > video > frame.
        """),
        epilog=textwrap.dedent("""\
            Examples:
        """)
    )

    parser.add_argument(
        "-c", "--camera",
        metavar="DEVICE",
        default=None,
        help="Camera device index (e.g. 0, 1) or device path (e.g. /dev/video0). "
            "Streams continuously and overlays --image onto any detected display."
    )
    parser.add_argument(
        "-v", "--video",
        metavar="FILE",
        default=None,
        help="Path to a video file to stream (e.g. clip.mp4). "
            "Overlays --image onto any detected display in each frame."
    )
    parser.add_argument(
        "-f", "--frame",
        metavar="FILE",
        default=None,
        help="Path to a single image file to process (e.g. frame.png). "
            "Detects a display in the image and overlays --image onto it."
    )
    parser.add_argument(
        "-i", "--image",
        metavar="FILE",
        default=None,
        help="Image to insert onto the detected display surface (e.g. slide.png). "
            "If omitted, defaults to default.jpg."
    )


    args = parser.parse_args()

    if not any([args.camera, args.video, args.image, args.frame]):
        parser.error("[ERROR] At least one argument required: -c DEVICE, -v FILE or -f FILE")

    return args


def open_video_capture(source):
    try:
        source = int(source)
    except ValueError:
        pass  # keep as string path

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open Video Capture: {source}")
        exit(1)

    print(f"[INFO] Video Capture opened: {source}")
    return cap



def stream(cap, image_to_insert):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Display', 1280, 720)
    alpha = 0.3
    smoothed_corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                print("[INFO] End of video")
            else:
                print("[ERROR] Failed to grab frame")
            break

        corners = display_detection.find_display(frame, debug=False)

        if corners is None:
            print("[ERROR] Display not found!")
            continue
        else:
            print("[INFO] Corners (TL, TR, BR, BL):")
            for p in corners.astype(int):
                print(tuple(p))

        # Smooth corners using EMA
        if smoothed_corners is None:
            smoothed_corners = corners.astype(np.float32)
        else:
            smoothed_corners = alpha * corners + (1 - alpha) * smoothed_corners

        display = image_warping.insert_image(frame, image_to_insert, smoothed_corners)
        cv2.imshow('Display', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.image:
        image_to_insert = cv2.imread(args.image)
    else:
        image_to_insert = cv2.imread('default.jpg')

    if args.camera or args.video:
        source = args.camera or args.video
        cap = open_video_capture(source)
        stream(cap, image_to_insert)
            

    elif args.frame:
        img = cv2.imread(args.frame)
        corners = display_detection.find_display(img, debug=False)

        if corners is None:
            print("[ERROR] Display not found")
            exit(1)
        else:
            print("[INFO] Corners (TL, TR, BR, BL):")
            for p in corners.astype(int):
                print(tuple(p))

        display = image_warping.insert_image(img, image_to_insert, corners)

        cv2.namedWindow('Display', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Display', 1280, 720)
        cv2.imshow('Display', display)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__  == "__main__":
    main()


