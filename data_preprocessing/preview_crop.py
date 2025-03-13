import cv2
import argparse

scale = 0.6

def get_cutout(input, square_size=(512, 512), x_start=0, y_start=0, frame_number=0):
    # Ensure x and y coordinates are not negative
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0

    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input}")
        return
    
    # Set the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return

    # Find the dimensions of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame dimensions: {frame_width}x{frame_height}")

    # if crop are is out of bounds, force it to be within the bounds
    if x_start + square_size[0] > frame_width:
        x_start = frame_width - square_size[0]
    if y_start + square_size[1] > frame_height:
        y_start = frame_height - square_size[1]

    # Drawing the rectangle
    top_left = (x_start, y_start)
    bottom_right = (x_start + square_size[0], y_start + square_size[1])
    color = (0, 0, 255)
    thickness = 2 
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    # Rezise the image to fit the screen
    resized_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))

    # Display the image with the crop area marked
    cv2.imshow('Preview Crop Area', resized_frame)
    cv2.waitKey(0)  # Close the window when any key is pressed
    cv2.destroyAllWindows()

    cap.release()

parser = argparse.ArgumentParser(description="Preview the crop area of a video.")
parser.add_argument("input", help="Input video file dir.")
parser.add_argument("-c", type=int, nargs=2, default=(512, 512), metavar=('width', 'height'), help="Shape of the crop area (width height).")
parser.add_argument("-x", type=int, default=430, help="X-coordinate of the top left corner of the crop.")
parser.add_argument("-y", type=int, default=256, help="Y-coordinate of the top left corner of the crop.")
parser.add_argument("-f", type=int, default=0, metavar=('Frame'), help="Frame number to preview.")

args = parser.parse_args()

get_cutout(args.input, tuple(args.c), args.x, args.y, args.f)
