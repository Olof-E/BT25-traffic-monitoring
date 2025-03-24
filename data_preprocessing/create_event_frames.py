import argparse
import time
import cv2
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import event_streamer_c

cuda_device = 0
decay_rate = 0.0002
frame_width = 640
frame_height = 480

decay = False  # set true to add decay to the input events

fps = 90


def bin_events(fpath, output_dir, clip_length, num_of_clips, bin_size, save_vids):
    torch.cuda.set_device(device=cuda_device)

    # stream = EventStream(fpath)
    read_from = 239
    last_time_high = 0
    event_buffer, read_from, last_time_high = event_streamer_c.c_fill_event_buffer(
        2_000, read_from, last_time_high
    )
    df_timestamps = pd.read_csv("timestamps.csv")

    time_windows0 = df_timestamps.iloc[:, 0].to_numpy()
    time_windows1 = df_timestamps.iloc[:, 1].to_numpy()
    time_windows = time_windows1 - time_windows1[0]
    time_windows10 = time_windows0 - time_windows0[0]
    print(time_windows10)
    print(time_windows)

    total_runtime = 0
    event_idx = 0
    last_time_window = 0

    curr_evt = event_buffer[event_idx]
    event_idx += 1

    for clip_nr in range(num_of_clips):
        frames = []

        start_time = time.time()
        with tqdm(
            range(clip_length * fps),
            ncols=86,
            desc=f"processing clip {clip_nr+1}/{num_of_clips}",
            leave=False,
            mininterval=0.25,
            miniters=50,
            unit=" frames",
        ) as t_proc:
            for i in t_proc:
                start = time_windows10[last_time_window]
                end = time_windows10[last_time_window + 1]
                last_time_window += 1

                frame = np.zeros((frame_height, frame_width))
                while curr_evt and (curr_evt.timestamp - time_windows0[0]) < end:
                    decay_multiplier = 1
                    if decay:  # add decay rate if desired
                        time_since_start = curr_evt.timestamp - start
                        decay_multiplier = np.exp(-(time_since_start * decay_rate))

                    frame[curr_evt.y, curr_evt.x] = curr_evt.polarity * decay_multiplier

                    if event_idx >= len(event_buffer):
                        event_buffer, read_from, last_time_high = (
                            event_streamer_c.c_fill_event_buffer(2_000, read_from, last_time_high)
                        )
                        event_idx = 0
                    curr_evt = event_buffer[event_idx]
                    event_idx += 1

                if not curr_evt:
                    print("Stream end reached. current clip processing aborted...")
                    exit(0)

                frames.append(torch.tensor(frame).to_sparse())

        print(f"Clip {clip_nr+1}/{num_of_clips}: saving...", end="\r")

        filename = f"{output_dir}event_frames_{clip_nr}.pt"
        torch.save(torch.stack(frames), filename)

        if save_vids:
            out = cv2.VideoWriter(
                f"{output_dir}event_clip_{clip_nr}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
                False,
            )
            shape = frames[0].shape
            with tqdm(
                range(len(frames)),
                ncols=86,
                desc=f"saving clip {clip_nr+1}/{num_of_clips}",
                leave=False,
                mininterval=0.25,
                miniters=50,
                unit=" frames",
            ) as t_save:
                for i in t_save:
                    image_scaled = minmax_scale(
                        frames[i].to_dense().numpy().ravel(), feature_range=(0, 255), copy=False
                    ).reshape(shape)
                    out.write(np.uint8(image_scaled))

            out.release()

        print(end="\x1b[2K")
        proc_time = time.time() - start_time
        print(
            f"Clip {clip_nr+1}/{num_of_clips} [\x1b[92m\u2714\x1b[0m] Finished in {time.strftime('%Mm %Ss', time.gmtime(proc_time))}\n"
        )

        total_runtime += proc_time

    print("==========================================")
    print(f"Processing finished in {time.strftime('%Mm %Ss', time.gmtime(total_runtime))}")
    print(f"Data saved to {output_dir}")


parser = argparse.ArgumentParser(
    description="A script that reads an EVT2 file as an Event Stream and bins events into frames of given size",
    usage="%(prog)s <path/to/event_file> <path/to/output_dir/> [options]",
)

parser.add_argument("filename", help="The path to the file containing the EVT2 RAW event data")
parser.add_argument(
    "output_dir", help="The path to the directory where the processed data should be saved"
)
parser.add_argument(
    "-n",
    "--clips_count",
    default=1,
    type=int,
    help="The number of videos/clips to cut the event frames into (default: %(default)s)",
)
parser.add_argument(
    "-l",
    "--length",
    default=60,
    type=int,
    help="The desired length of the clips in seconds (default: %(default)s s)",
)
parser.add_argument(
    "-b",
    "--bin",
    default=10.0,
    type=float,
    help="The desired bin size for the event frames in milliseconds (default: %(default)s ms)",
)
parser.add_argument(
    "--save-vid",
    action="store_true",
    default=False,
    help="Can be set to also generate .mp4 files of the processed event frames (default: %(default)s)",
)

args = parser.parse_args()

bin_events(args.filename, args.output_dir, args.length, args.clips_count, args.bin, args.save_vid)
