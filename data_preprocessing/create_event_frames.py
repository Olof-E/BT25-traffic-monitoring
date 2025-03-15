import argparse
import time
import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale
import torch
from tqdm import tqdm

from event_streamer import EventStream

cuda_device = 0
decay_rate = 0.0002
frame_width = 640
frame_height = 480

decay = False  # set true to add decay to the input events


def bin_events(fpath, output_dir, clip_length, num_of_clips, bin_size, save_vids):
    torch.cuda.set_device(device=cuda_device)

    stream = EventStream(fpath)

    total_runtime = 0

    for clip_nr in range(num_of_clips):
        frames = []

        curr_evt = stream.read()
        start = curr_evt.timestamp

        with tqdm(
            range(clip_length * 100),
            ncols=86,
            desc=f"processing clip {clip_nr+1}/{num_of_clips}",
            leave=False,
            unit=" frames",
        ) as t_proc:
            for _ in t_proc:
                frame = np.zeros((frame_height, frame_width))
                while curr_evt and (curr_evt.timestamp - start) < bin_size * 1000:
                    time_since_start = curr_evt.timestamp - start

                    decay_multiplier = 1
                    if decay:  # add decay rate if desired
                        decay_multiplier = np.exp(-(time_since_start * decay_rate))
                    frame[curr_evt.y, curr_evt.x] = curr_evt.polarity * decay_multiplier

                    curr_evt = stream.read()

                if not curr_evt:
                    print("Stream end reached. current clip processing aborted...")
                    exit(0)

                start = curr_evt.timestamp
                frames.append(torch.tensor(frame).to_sparse())

        print(f"Clip {clip_nr+1}/{num_of_clips}: saving...", end="\r")

        filename = f"{output_dir}event_frames_{clip_nr}.pt"
        torch.save(torch.stack(frames), filename)

        if save_vids:
            out = cv2.VideoWriter(
                f"{output_dir}event_clip_{clip_nr}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                100,
                (frame_width, frame_height),
                False,
            )
            shape = frames[0].shape
            with tqdm(
                range(len(frames)),
                ncols=86,
                desc=f"saving clip {clip_nr+1}/{num_of_clips}",
                leave=False,
                unit=" frames",
            ) as t_save:
                for i in t_save:
                    image_scaled = minmax_scale(
                        frames[i].to_dense().numpy().ravel(), feature_range=(0, 255), copy=False
                    ).reshape(shape)
                    out.write(np.uint8(image_scaled))

            out.release()

        print(end="\x1b[2K")
        proc_time = time.strftime(
            "%Mm %Ss", time.gmtime(t_proc.format_dict["elapsed"] + t_save.format_dict["elapsed"])
        )
        print(f"Clip {clip_nr+1}/{num_of_clips} [\x1b[92m\u2714\x1b[0m] Finished in {proc_time}\n")

        total_runtime += t_proc.format_dict["elapsed"] + t_save.format_dict["elapsed"]

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
    default=10,
    type=int,
    help="The desired bin size for the event frames in milliseconds (default: %(default)s ms)",
)
parser.add_argument(
    "--save_vid",
    action="store_true",
    default=False,
    help="Can be set to also generate .mp4 files of the processed event frames (default: %(default)s)",
)

args = parser.parse_args()

bin_events(args.filename, args.output_dir, args.length, args.clips_count, args.bin, args.save_vid)
