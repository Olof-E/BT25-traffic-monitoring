import sys
import time
import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale
import torch
from tqdm import tqdm

from event_streamer import EventStream

cuda_device = 0
num_of_clips = 10
decay_rate = 0.0002
frame_width = 640
frame_height = 480

decay = False  # set true to add decay to the input events

output_dir = "../clips/events/"


torch.cuda.set_device(device=cuda_device)

stream = EventStream("../events.raw")

total_runtime = 0

for clip_nr in range(num_of_clips):
    print(f"")
    frames = []
    event_index = 0

    curr_evt = stream.read()
    start = curr_evt.timestamp

    with tqdm(
        range(6000),
        ncols=80,
        desc=f"clip {clip_nr+1}/{num_of_clips}",
        leave=False,
        unit=" frames",
    ) as t_proc:
        for _ in t_proc:
            frame = np.zeros((frame_height, frame_width))
            while curr_evt and (curr_evt.timestamp - start) < 10_000:
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

    # Review outputs
    filename = f"{output_dir}event_frames_{clip_nr}.pt"
    torch.save(torch.stack(frames), filename)
    # print(f"Saved raw event frames to: {filename}")

    out = cv2.VideoWriter(
        f"{output_dir}event_clip_{clip_nr}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        100,
        (frame_width, frame_height),
        False,
    )
    # print(f"Saved event video to: {filename}")

    shape = frames[0].shape
    with tqdm(
        range(len(frames)),
        ncols=80,
        desc=f"clip {clip_nr+1}/{num_of_clips}",
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
        "%M m %S s", time.gmtime(t_proc.format_dict["elapsed"] + t_save.format_dict["elapsed"])
    )
    print(f"Clip {clip_nr+1}/{num_of_clips} [\x1b[92m\u2714\x1b[0m] Finished in {proc_time}")

    total_runtime += t_proc.format_dict["elapsed"] + t_save.format_dict["elapsed"]

print("==========================================")
print(f"Processing finished in {time.strftime('%M m %S s', time.gmtime(total_runtime))}")
print(f"Data saved to {output_dir}")
