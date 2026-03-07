import display_detection
import image_warping
import cv2
import argparse
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Camera capture tool
            -------------------
            Streams from a camera device in real-time.
            Optionally saves a single frame as a PNG image.
            Optionally loads an existing image for display.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python main.py -c 0                        # stream from camera 0
              python main.py -c 0 -f frame.png           # stream and save a frame
              python main.py -i photo.png                # display an existing image
              python main.py -i photo.png -f out.png     # load and re-save a frame
        """)
    )

    parser.add_argument(
        "-c", "--camera",
        metavar="DEVICE",
        default=None,
        help="Camera device index (e.g. 0, 1) or device path (e.g. /dev/video0)"
    )
    parser.add_argument(
        "-f", "--frame",
        metavar="FILE",
        default=None,
        help="(Optional) Save a captured frame to this PNG file (e.g. frame.png)"
    )
    parser.add_argument(
        "-i", "--image",
        metavar="FILE",
        default=None,
        help="(Optional) Load and display an existing image file (e.g. photo.png)"
    )

    args = parser.parse_args()

    if not args.camera and not args.frame and not args.image:
        parser.error("At least one argument required: -c DEVICE, -f FILE, and/or -i FILE")

    return args

def open_camera(device):
    try:
        device = int(device)
    except ValueError:
        pass  # keep as string path

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {device}")
        exit(1)

    print(f"[INFO] Camera opened: {device}")
    return cap

def stream(cap, image_to_insert):
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        
        img = cv2.imread(frame)
        corners = display_detection.find_display(img, debug=False)

        if corners is None:
            print("Экран не найден")
            exit(1)
        else:
            print("Corners (TL, TR, BR, BL):")
            for p in corners.astype(int):
                print(tuple(p))

        display = image_warping.insert_image(image_to_insert, corners)

        cv2.namedWindow('Display')
        cv2.imshow('Display', display)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.image:
        image_to_insert = args.image
    else:
        image_to_insert = cv2.imread('default.jpg')

    if args.camera:
        cap = open_camera(args.camera)
        stream(cap, image_to_insert)

    elif args.frame:
        img = cv2.imread(args.frame)
        corners = display_detection.find_display(img, debug=False)

        if corners is None:
            print("Экран не найден")
            exit(1)
        else:
            print("Corners (TL, TR, BR, BL):")
            for p in corners.astype(int):
                print(tuple(p))

        display = image_warping.insert_image(img, image_to_insert, corners)

        cv2.namedWindow('Display')
        cv2.imshow('Display', display)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        exit(1)

    return 1


if __name__  == "__main__":
    main()


