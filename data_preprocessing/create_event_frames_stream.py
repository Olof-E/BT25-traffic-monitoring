import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale
import torch
from tqdm import tqdm

from event_streamer import EventStream

cuda_device = 0
start_video_nr = 0
end_video_nr = 1
decay_rate = 1
frame_width = 640
frame_height = 480

decay = False  # set true to add decay to the input events

torch.cuda.set_device(device=cuda_device)

for video_nr in range(start_video_nr, end_video_nr):

    frames = []
    event_index = 0

    stream = EventStream("../events.raw")

    curr_evt = stream.read()
    start = curr_evt.timestamp

    for _ in tqdm(range(15_000)):
        frame = np.zeros((frame_height, frame_width))

        while (curr_evt[3] - start) < 10_000:
            time_since_start = curr_evt[3] - start

            decay_multiplier = 1
            if decay:  # add decay rate if desired
                decay_multiplier = np.exp(-time_since_start / decay_rate)
            frame[curr_evt[1], curr_evt[0]] = curr_evt[2] * decay_multiplier

            curr_evt = stream.read()

        start = curr_evt[3]
        frames.append(frame)

    print("Processing complete.")

    # # Review outputs
    # filename = f"event_frames_{video_nr}.pt"
    # print(frames.shape)
    # torch.save(frames, filename)
    # print(f"Saved frames to: {filename}")

    # print(frames.size)
    # print(frames.shape)

    out = cv2.VideoWriter(
        f"testVid_stream.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 100, (frame_width, frame_height), 0
    )

    shape = frames[0].shape
    for i in tqdm(range(len(frames))):
        image_scaled = minmax_scale(frames[i].ravel(), feature_range=(0, 255)).reshape(shape)
        out.write(np.uint8(image_scaled))

    out.release()
