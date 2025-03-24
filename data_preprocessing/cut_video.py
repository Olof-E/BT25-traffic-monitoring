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
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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
        miniters=50,
    ) as t:
        start_time = time.time()
        for _ in t:
            ret, frame = cap.read()
            if not ret:
                break

            new_frame = cv2.resize(
                src=frame, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA
            )
            frames.append(new_frame)

            if len(frames) == frames_per_clip:
                if save_thread and save_thread.is_alive():
                    save_thread.join()

                proc_time = time.time() - start_time
                start_time = time.time()

                save_thread = Thread(
                    target=save_clip,
                    args=[frames.copy(), output_dir, current_clip, num_of_clips, fps, proc_time],
                )
                save_thread.start()

                frames.clear()

                current_clip += 1
                t.set_description(f"processing clip {current_clip+1}/{num_of_clips}")

    cap.release()
    if save_thread and save_thread.is_alive():
        save_thread.join()


def save_clip(frames, output_dir, current_clip, num_of_clips, fps, proc_time):
    start_time = time.time()
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
    save_time = time.time() - start_time

    tqdm.write(
        f"Clip {current_clip+1}/{num_of_clips} [\x1b[92m\u2714\x1b[0m] Finished in \x1b[1m{time.strftime('%Mm %Ss', time.gmtime(proc_time+save_time))}\x1b[22m\n"
    )


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

start_time = time.time()
split_video(args.filename, args.output_dir, args.length, args.clips_count)

total_runtime = time.time() - start_time

print("=================================================")
print(f"\nFinished in \x1b[1m{time.strftime('%Mm %Ss', time.gmtime(total_runtime))}\x1b[22m")
print(f"Clips saved to: \x1b[1m{args.output_dir}\x1b[22m")

# yolo track model="yolo/yolo11n.pt" source="2-08/_8.mp4" conf=0.3, iou=0.35 project="yolo/results/" save_txt=true device="cuda:0" batch=64 half verbose=False


# yolo track model="yolo/yolo12n.pt" source="2-08/_0-area.mp4" conf=0.3, iou=0.8 project="yolo/results/" save_txt=False device="cuda:0" batch=128 half verbose=False stream_buffer=True augment=True agnostic_nms=True


# yolo track model="yolo/yolo12s.pt" source="2-08/_2.mp4" conf=0.3, iou=0.7 project="yolo/results/" save_txt=True device="cuda:0" batch=128 half verbose=False stream_buffer=False augment=False agnostic_nms=False tracker="yolo/botsort.yaml"
