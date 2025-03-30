import cv2
from matplotlib.animation import FFMpegWriter
from sklearn.preprocessing import minmax_scale
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
nr = 4

visualize = True

# 0 - Person
# 1 - Bicycle
# 2 - Car
# 5 - Bus
# 7 - Truck

tracked_classes = ("0", "1", "2", "5", "7")


def get_targets(directory, target_length, file_nr):
    target_tensors = np.empty(target_length, dtype=object)

    for filename in sorted(filter(lambda f: f.endswith(".txt"), os.listdir(directory))):
        with open(os.path.join(directory, filename)) as file:
            filename = filename.split("_")
            if int(filename[1]) != file_nr:
                continue
            target_data = [
                float(value)
                for line in file
                if line.strip().split()[0] in tracked_classes
                for value in line.split()[0:5]
            ]
            target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 5)
            target_tensors[int(filename[2].rsplit(".", maxsplit=1)[0]) - 1] = target_tensor

    return target_tensors


# scp -r devbox:/home/olofeli/traffic-monit/data/clips/w31/2-08 ./


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


def annotate_frame(frame, targets, frame_idx, overlay, out):
    global max_value

    x_min, y_min, x_max, y_max = (25, 280, 225, 480)
    small_frame_dim = 64
    G_overlay = np.zeros((small_frame_dim, small_frame_dim))

    for t in range(len(targets)):
        class_type = targets[t][0]
        center_x, center_y = targets[t][1] - [25, 280]
        w, h = targets[t][2]

        x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
        x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]

        # Create and add rectangle patch and mark circle for bbox
        # if visualize:
        #     rect = patches.Rectangle(
        #         (x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none"
        #     )
        #     cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
        #     axes[0].add_patch(rect)
        #     axes[0].add_patch(cirk)

        # Scale down the values to the 64x64 frame
        w_small = w * 64 / 200
        h_small = h * 64 / 200
        center_x_small = center_x * 64 / 200
        center_y_small = center_y * 64 / 200

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
    x_min, y_min, x_max, y_max = (25, 280, 225, 480)
    if visualize:
        overlay2 = cv2.resize(
            overlay.detach().clone().numpy(), dsize=(200, 200), interpolation=cv2.INTER_NEAREST
        )
        tempframe = cv2.resize(
            np.concatenate(
                (
                    frame.detach().clone() * 255,
                    np.ones((200, 20)) * 255,
                    overlay2 / max(np.max(overlay2), 0.00001) * 255,
                ),
                axis=1,
            ),
            dsize=(420 * 2, 200 * 2),
            interpolation=cv2.INTER_NEAREST,
        )
        # image_scaled = cv2.normalize(
        #     tempframe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        # )
        # print(repr(image_scaled.astype(np.uint8)))
        out.write(tempframe.astype(np.uint8))

    return [
        frame.detach().clone(),
        overlay.detach().clone(),
    ], overlay


H = transform.SimilarityTransform()
H.params = np.array(
    [
        [8.50065481e-01, 7.29692005e-03, 1.22287204e02],
        [-7.29692005e-03, 8.50065481e-01, 6.95716137e01],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

for i in range(1, 6 + 15):
    print(f"Processing data: {i}")
    targets = get_targets(
        f"../w31/box2/2-07-31/track/labels/", 5400 * 30, i
    )  # The hardcoded number causes a diff in the long run, needs to look into.
    frames_tensor = []
    label_tensor = []
    new_label = []
    data = torch.load(f"../w31/box2/2-07-31/events/event_frames_{i}.pt")
    small_frame_dim = 64
    max_value = 0
    overlay = torch.zeros((small_frame_dim, small_frame_dim))
    out = None
    if visualize:
        out = cv2.VideoWriter(
            filename="label-test-anim.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=90,
            frameSize=(420 * 2, 200 * 2),
            isColor=False,
        )

    for frame_idx in tqdm(
        range(int(len(data))),
        desc=f"annotating frames",
        ncols=86,
        mininterval=0.25,
    ):
        frame = data[frame_idx].to_dense()[280:480, 25:225]  # [180:480, 100:400]
        warped = []

        if targets[frame_idx] == None:
            print("missing:", frame_idx)
            res = [
                frame.detach().clone(),
                torch.zeros((64, 64)),
            ]

        elif len(targets[frame_idx]) == 0:
            res = [
                frame.detach().clone(),
                torch.zeros((64, 64)),
            ]

        else:
            for tar in range(len(targets[frame_idx])):
                x, y = targets[frame_idx][tar][1:3] * torch.tensor([736, 460])
                w, h = targets[frame_idx][tar][3:] * torch.tensor([736, 460])

                x_min, y_min = np.array([x, y]) - [w / 2, h / 2]
                x_max, y_max = np.array([x, y]) + [w / 2, h / 2]

                x_min, y_min = H._apply_mat((x_min, y_min), H.inverse.params)[0]
                x_max, y_max = H._apply_mat((x_max, y_max), H.inverse.params)[0]
                w = x_max - x_min
                h = y_max - y_min

                transformed_coordinates = H._apply_mat((x, y), H.inverse.params)[0]
                warped.append(
                    [targets[frame_idx][tar][0], transformed_coordinates, torch.tensor([w, h])]
                )

            res, overlay = annotate_frame(frame, warped, frame_idx, overlay, out)

        frames_tensor.append(res[0])
        label_tensor.append(res[1])

    print("\nSaving data...")
    if visualize:
        out.release()
    torch.save(
        [torch.stack(frames_tensor).to_sparse(), torch.stack(label_tensor).to_sparse()],
        f"../clips/frames_with_labels/{i-6}.pt",
    )
    exit()
