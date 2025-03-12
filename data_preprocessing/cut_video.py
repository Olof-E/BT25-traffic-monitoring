import sys
import cv2
from tqdm import tqdm
from threading import Thread

video_length = 60  # in seconds
input_file = "../srcVid.mp4"  # "/mnt/usb_data_READ_ONLY/data_collection/week_31/box_1/2024_07_30_16_11_37_recordings/video.avi" # video file to divide
output_dir = "../clips/test/"  # "/home/olofeli/traffic-monit/data/clips/" # output directory to save the files.

square_size = (512, 512)  # size of the cutout
x_start = 430
y_start = square_size[1] - 256  # 256 approximated case by case


def split_video(path, length, output):

    cap = cv2.VideoCapture(path)
    fps = 100  # int(cap.get(cv2.CAP_PROP_FPS))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_clip = length * fps
    current_clip = 0
    frames = []

    # total_clips = int(np.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)/frames_per_clip))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(
            frame[y_start : y_start + square_size[1], x_start : x_start + square_size[0]].copy()
        )

        if len(frames) == frames_per_clip:
            Thread(target=save_clip, args=[frames.copy(), output, current_clip, fps]).start()

            frames.clear()

            current_clip += 1
            # print(f"Video: {current_clip} is processed")

    cap.release()


def save_clip(frames, output, current_clip, fps):
    out = cv2.VideoWriter(
        f"{output}_{current_clip}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        square_size,
    )
    for f in frames:
        out.write(f)

    out.release()


print(sys.argv[1:])
split_video(input_file, video_length, output_dir)


# yolo track model="yolo/yolo11n.pt" source="clips/test/" conf=0.25, iou=0.5 project="yolo/results/" save_txt=true device="cuda:0"
