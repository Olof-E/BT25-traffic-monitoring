import argparse
import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

"""
- read in the event frames
- unfold, restructure and fold them randomly
- create target frames with 0
"""


def shuffle_frame(frame):

    frame_shape = np.zeros((frame.shape[1], frame.shape[0]))

    # unfold tensor
    u = F.unfold(frame.unsqueeze(0), kernel_size=16, stride=16, padding=0)

    # Shape is [1, 4096, 16], permute last dimension
    permuted_indices = torch.randperm(u.shape[-1])
    pu = u[:, permuted_indices]

    # Fold shuffled blocks back to an image
    shuffled_frame = F.fold(pu, output_size=(256, 256), kernel_size=16, stride=16, padding=0)
    # plt.imshow(shuffled_frame.numpy()[0], cmap="gray")
    # plt.show()

    return [shuffled_frame[0], torch.zeros((64, 64))]


def create_noise(input_dir, output_dir, num_clips, save_name_idx):

    nr_noise_frames = 5400
    rois = np.load(os.path.join(input_dir, "rois.npy"))

    zero_256_frame = np.full((256, 256), 0, dtype=np.uint8)
    divider_256_frame = np.full((256, 20), 255, dtype=np.uint8)

    for i in range(num_clips):
        frames_tensor = []
        label_tensor = []
        data = torch.load(os.path.join(input_dir, f"events/event_frames_{i}.pt"))
        # w38/box3/3-09-27/events/event_frames_6.pt
        # w31/box2/2-08-01/events/event_frames_14.pt

        for j, roi in enumerate(rois):
            out = cv2.VideoWriter(
                filename=os.path.join(output_dir, f"{i+save_name_idx}-{j}-vis.mp4"),
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=90,
                frameSize=(
                    256 * 5 + 20 * 4,
                    256,
                ),
                isColor=False,
            )
            for frame_inx in tqdm(
                range(nr_noise_frames),
                desc=f"clip {i+1}/{num_clips} ROI {j+1}/{len(rois)}",
                ncols=86,
                mininterval=0.25,
                leave=False,
            ):
                frame = data[frame_inx].to_dense()[roi[1] : roi[3], roi[0] : roi[2]]
                res = shuffle_frame(frame)

                frames_tensor.append(res[0])
                label_tensor.append(res[1])

                combined_frames = np.concatenate(
                    (
                        res[0].numpy() * 255,
                        divider_256_frame,
                        zero_256_frame,
                        divider_256_frame,
                        zero_256_frame,
                        divider_256_frame,
                        zero_256_frame,
                        divider_256_frame,
                        zero_256_frame,
                    ),
                    axis=1,
                )
                out.write(np.uint8(combined_frames))

            torch.stack(label_tensor).to_sparse()
            torch.save(
                [
                    torch.stack(frames_tensor).to_sparse(),
                    torch.stack(label_tensor).to_sparse(),
                    torch.stack(label_tensor).to_sparse(),
                    torch.stack(label_tensor).to_sparse(),
                    torch.stack(label_tensor).to_sparse(),
                ],
                os.path.join(output_dir, f"{i+save_name_idx}-{j}.pt"),
            )
            out.release()


parser = argparse.ArgumentParser(
    description="A script that creates noise training data",
    usage="%(prog)s <path/to/input_dir/> <path/to/output_dir/> [options]",
)

parser.add_argument(
    "input_dir",
    help="The path to the top-level directory containing the data, i.e rois, events, normal",
)
parser.add_argument(
    "output_dir", help="The path to the directory where the processed data should be saved"
)
parser.add_argument(
    "-n",
    "--clips_count",
    default=1,
    type=int,
    help="The number of videos/clips to create (default: %(default)s)",
)
parser.add_argument(
    "--save-name",
    required=True,
    type=int,
    help="Can be set to also generate .mp4 files of the processed data (Required)",
)
parser.add_argument(
    "--save-vid",
    action="store_true",
    default=False,
    help="Can be set to also generate .mp4 files of the processed data (default: %(default)s)",
)


args = parser.parse_args()

create_noise(args.input_dir, args.output_dir, args.clips_count, args.save_name)
