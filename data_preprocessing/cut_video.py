import cv2
from tqdm import tqdm
from threading import Thread

video_length = 60  # in seconds
fps = 100
input_file = "../srcVid.mp4"  # "/mnt/usb_data_READ_ONLY/data_collection/week_31/box_1/2024_07_30_16_11_37_recordings/video.avi" # video file to divide
output_dir = "../clips/test/"  # "/home/olofeli/traffic-monit/data/clips/" # output directory to save the files.

new_width = 0
new_height = 0


def split_video(path, length, output):

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    aspect_ratio = height / width
    new_width = 720
    new_height = int(new_width * aspect_ratio)

    frames_per_clip = length * fps
    current_clip = 0
    frames = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        new_frame = cv2.resize(src=frame, dsize=(new_width, new_height))
        # print(new_frame)
        frames.append(new_frame.copy())

        if len(frames) == frames_per_clip:
            Thread(target=save_clip, args=[frames.copy(), output, current_clip, fps]).start()

            frames.clear()

            current_clip += 1

    cap.release()


def save_clip(frames, output, current_clip, fps):
    out = cv2.VideoWriter(
        f"{output}_{current_clip}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
        False,
    )
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))

    print(f"Video: {current_clip} is processed")
    out.release()


split_video(input_file, video_length, output_dir)


# yolo track model="yolo/yolo11n.pt" source="clips/test/" conf=0.35, iou=0.5 project="yolo/results/" save_txt=true device="cuda:0"
