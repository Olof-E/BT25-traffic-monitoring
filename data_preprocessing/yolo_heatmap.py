import cv2
import argparse
from ultralytics import solutions

def main(input_video):
    cap = cv2.VideoCapture(input_video)
    assert cap.isOpened(), "Error reading video file"

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize heatmap object
    heatmap = solutions.Heatmap(
        show=True,  # display the output
        model="/home/tobias/Documents/yolo11n.pt",  # path to the YOLO11 model file
        colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
        # classes=[0, 1, 2, 3, 5, 7],  # generate heatmap for all traffic classes
        classes=[0], # generate heatmap for people only
        conf=0.4,
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = heatmap(im0)

        # print(results)  # access the output

        video_writer.write(results.plot_im)  # write the processed frame.

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a heatmap for a video using YOLO.")
    parser.add_argument("input_video", help="Path to the input video file.")
    args = parser.parse_args()

    main(args.input_video)