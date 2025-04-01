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


def create_noise(frame):

    frame_shape = np.zeros((frame.shape[1], frame.shape[0]))

    # unfold tensor
    u = F.unfold(torch.tensor(frame).unsqueeze(0), kernel_size=16, stride=16, padding=0)

    # Shape is [1, 4096, 16], permute last dimension
    permuted_indices = torch.randperm(u.shape[-1])
    pu = u[:, permuted_indices]

    # Fold shuffled blocks back to an image
    shuffled_frame = F.fold(pu, output_size=(250, 250), kernel_size=16, stride=16, padding=0)

    return [torch.tensor(shuffled_frame), torch.tensor(np.zeros((64, 64)))]


i = 0
j = 280
nr = 1
nr_noise_frames = 3000
frames_tensor = []
label_tensor = []
data = torch.load("csv4_frames.pt")

for frame_inx in tqdm(range(nr_noise_frames)):
    x_min, y_min, x_max, y_max = (90, 50, 290, 250)
    frame = data[frame_inx].to_dense()[y_min:y_max, x_min:x_max]
    res = create_noise(frame)

    if frame_inx % 500 == 0:
        print(frame_inx)

    frames_tensor.append(res[0])
    label_tensor.append(res[1])

torch.save(
    [torch.stack(frames_tensor).to_sparse(), torch.stack(label_tensor).to_sparse()],
    f"frames_with_labels/{nr}_noise.pt",
)
