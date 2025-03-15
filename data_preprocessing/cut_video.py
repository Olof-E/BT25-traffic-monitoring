import time
import cv2
import argparse
from tqdm import tqdm
from threading import Thread

# "/mnt/usb_data_READ_ONLY/data_collection/week_31/box_2/2024_08_01_16_40_03_recordings/video.avi"  # video file to divide
# "/home/olofeli/traffic-monit/data/clips/w31/2-08/"


def split_video(fpath, output_dir, length, num_of_clips):

    cap = cv2.VideoCapture(fpath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = 100

    aspect_ratio = height / width
    new_width = 736
    new_height = int(new_width * aspect_ratio)

    frames_per_clip = length * fps
    current_clip = 0
    frames = []

    save_thread = None

    with tqdm(
        range(frames_per_clip * num_of_clips),
        desc=f"processing clip {current_clip+1}/{num_of_clips}",
        ncols=86,
        position=0,
        leave=False,
        mininterval=0.25,
    ) as t:
        for _ in t:
            ret, frame = cap.read()
            if not ret:
                break

            new_frame = cv2.resize(src=frame, dsize=(new_width, new_height))
            frames.append(new_frame.copy())

            if len(frames) == frames_per_clip:
                if save_thread and save_thread.is_alive():
                    save_thread.join()

                save_thread = Thread(
                    target=save_clip,
                    args=[frames.copy(), output_dir, current_clip, num_of_clips, fps],
                )
                save_thread.start()

                frames.clear()

                current_clip += 1
                t.set_description(f"processing clip {current_clip+1}/{num_of_clips}")

    cap.release()
    if save_thread.is_alive():
        save_thread.join()
        tqdm.write(
            f"\nFinished in \x1b[1m{time.strftime('%Mm %Ss', time.gmtime(t.format_dict['elapsed']))}\x1b[22m"
        )
        tqdm.write(f"Clips saved to: \x1b[1m{output_dir}\x1b[22m")


def save_clip(frames, output_dir, current_clip, num_of_clips, fps):
    out = cv2.VideoWriter(
        f"{output_dir}_{current_clip}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for f in tqdm(
        frames,
        desc=f"saving clip {current_clip+1}/{num_of_clips}",
        ncols=86,
        mininterval=0.25,
        leave=False,
    ):
        out.write(f)

    out.release()


parser = argparse.ArgumentParser(
    description="A script that reads a video file and downscales and cuts it into a desired amount of clips and length",
    usage="%(prog)s <path/to/event_file> <path/to/output_dir/> [options]",
)

parser.add_argument("filename", help="The path to the source video file")
parser.add_argument("output_dir", help="The path to the directory where the clips should be saved")
parser.add_argument(
    "-n",
    "--clips_count",
    default=1,
    type=int,
    help="The number of videos/clips to cut the original video into (default: %(default)s)",
)
parser.add_argument(
    "-l",
    "--length",
    default=60,
    type=int,
    help="The desired length of the clips in seconds (default: %(default)s s)",
)

args = parser.parse_args()


split_video(args.filename, args.output_dir, args.length, args.clips_count)


# yolo track model="yolo/yolo11n.pt" source="clips/test/" conf=0.35, iou=0.5 project="yolo/results/" save_txt=true device="cuda:0" batch=16 half imgsz="(736, 416)"
