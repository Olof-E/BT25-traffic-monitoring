import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches
from skimage import transform
import os

""" 
1. Get the labels from yolo/result/labels
2. Map the coordinates with the event video

"""

torch.cuda.set_device(0)
nr = 0


def get_targets(directory, target_length):
    target_tensors = np.empty(target_length, dtype=object)

    for filename in sorted(filter(lambda f: f.endswith(".txt"), os.listdir(directory))):
        with open(os.path.join(directory, filename)) as file:
            target_data = [
                float(value)
                for line in file
                if line.strip().split()[0] in ("2", "5", "7", "61")
                for value in line.split()[1:5]
            ]
            target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 4)
            target_tensors[int(filename[3:-4]) - 1] = target_tensor

    return target_tensors


def count_events(frame, min_max):
    y_min, y_max, x_min, x_max = min_max
    number_of_events = torch.sum(frame > 0)
    min_events_threshold = max(
        1, 0.01 * ((x_max - x_min) * (y_max - y_min))
    )  # Dynamic minimum based on bbox area
    if number_of_events < min_events_threshold:
        ratio = 0
    else:
        ratio = (number_of_events - min_events_threshold) / ((x_max - x_min) * (y_max - y_min))
    return ratio


def leaky_integrator(input_data, tau, v):
    v_next = v + (1 / tau) * (input_data - v)
    return v_next


def annotate_frame(frame, targets, frame_idx, overlay):
    global max_value
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    x_min, y_min, x_max, y_max = (0, 50, 200, 250)
    axes[0].imshow(frame[y_min:y_max, x_min:x_max], cmap="gray")
    small_frame_dim = 64
    G_overlay = np.zeros((small_frame_dim, small_frame_dim))

    for t in range(len(targets)):
        center_x, center_y = targets[t][0][0], targets[t][0][1]
        w, h = targets[t][1][0], targets[t][1][1]
        # id = targets[t][4]
        # print(id)
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        x_max = center_x + w / 2
        y_max = center_y + h / 2

        # Create and add rectangle patch and mark circle for bbox
        rect = patches.Rectangle(
            (x_min - 54, y_min - 51), w, h, linewidth=1, edgecolor="r", facecolor="none"
        )
        cirk = patches.Circle(
            (center_x - 54, center_y - 51), radius=1, edgecolor="r", facecolor="red"
        )
        axes[0].add_patch(rect)
        axes[0].add_patch(cirk)

        # Scale down the values to the 64x64 frame
        w_small = w * 64 / 200
        h_small = h * 64 / 200
        center_x_small = center_x * 64 / 200 - 17  # hardcoded adjustment
        center_y_small = center_y * 64 / 200 - 16  # hardcoded adjustment

        # Adjust sigma proportionally for the smaller frame
        sigma_x_small = w_small / 6
        sigma_y_small = h_small / 6
        events_factor = count_events(
            frame[int(y_min) : int(y_max), int(x_min) : int(x_max)],
            (y_min, y_max, x_min, x_max),
        )

        # Create Gaussian mask on the smaller frame
        X, Y = np.meshgrid(
            np.linspace(0, small_frame_dim, small_frame_dim),
            np.linspace(0, small_frame_dim, small_frame_dim),
        )
        G = np.exp(
            -(
                (X - center_x_small) ** 2 / (2 * sigma_x_small**2)
                + (Y - center_y_small) ** 2 / (2 * sigma_y_small**2)
            )
        )
        G_normalized = G * events_factor

        G_overlay = np.maximum(G_overlay, G_normalized)

        # Apply to the small overlay

    if torch.max(G_overlay) > max_value:
        max_value = torch.max(G_overlay)

    overlay = leaky_integrator(
        G_overlay.detach().clone(),
        tau=10,
        v=overlay.detach().clone(),
    )
    axes[1].imshow(overlay, cmap="hot", extent=(0, 64, 64, 0), vmin=0, vmax=0.03)

    axes[0].axis("off")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"forlabeltransfer/{frame_idx}.png", bbox_inches="tight", pad_inches=0)

    plt.close(fig)

    x_min, y_min, x_max, y_max = (0, 50, 200, 250)
    return [
        frame[y_min:y_max, x_min:x_max].detach().clone(),
        overlay.detach().clone(),
    ], overlay


H = transform.estimate_transform(
    "euclidean",
    np.array([[28, 57], [462, 6], [607, 623], [54, 590], [248, 229], [522, 110]]) / 639,
    np.array([[9, 26], [211, 1], [278, 296], [9, 288], [113, 102], [234, 47]]) / 299,
)


targets = get_targets(
    f"../yolo/results/track/labels/", 27000
)  # The hardcoded number causes a diff in the long run, needs to look into.
frames_tensor = []
label_tensor = []
new_label = []
data = torch.load(f"../clips/events/event_frames_{nr}.pt")
small_frame_dim = 64
max_value = 0
overlay = torch.zeros((small_frame_dim, small_frame_dim))
x_min, y_min, x_max, y_max = (0, 180, 300, 480)

for frame_idx in tqdm(range(int(len(data)))):
    frame = data[frame_idx].to_dense()[y_min:y_max, x_min:x_max]
    warped = []

    if targets[frame_idx] == None:
        print("missing:", frame_idx)
        res = [frame[0:200, 0:200].detach().clone(), torch.zeros((64, 64))]

    elif len(targets[frame_idx]) == 0:
        res = [frame[0:200, 0:200].detach().clone(), torch.zeros((64, 64))]

    else:
        for tar in range(len(targets[frame_idx])):
            transformed_coordinates = np.dot(
                H, torch.concat((targets[frame_idx][tar][:2] * 299, torch.ones(1)))
            )
            transformed_coordinates = transformed_coordinates[:2] / transformed_coordinates[2]
            warped.append([transformed_coordinates, targets[frame_idx][tar][2:] * 299])

        res, overlay = annotate_frame(frame, warped, frame_idx, overlay)

    frames_tensor.append(res[0])
    label_tensor.append(res[1])


torch.save(
    [torch.stack(frames_tensor).to_sparse(), torch.stack(label_tensor).to_sparse()],
    f"../clips/frames_with_labels/{nr}.pt",
)
