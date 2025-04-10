from concurrent.futures import thread
from math import exp
from threading import Thread
import cv2
import torch
import numpy as np
from tqdm import tqdm
from skimage import transform
import os

""" 
1. Get the labels from yolo/result/labels
2. Map the coordinates with the event video

"""

torch.cuda.set_device(0)

# 0 - Person
# 1 - Bicycle
# 2 - Car
# 5 - Bus
# 7 - Truck

tracked_classes = [0, 2, 5, 7]


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
                if line.strip().split()[0] in str(tracked_classes)
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
        1, 0.004 * ((x_max - x_min) * (y_max - y_min))
    )  # Dynamic minimum based on bbox area
    if number_of_events < min_events_threshold:
        ratio = 0
    else:
        ratio = (number_of_events - min_events_threshold) / ((x_max - x_min) * (y_max - y_min))
    return ratio


def leaky_integrator(input_data, tau, v):
    v_next = v + (1 / tau) * (input_data - v)
    return v_next


def annotate_frame(frame, targets, overlays, out, roi, clip_maximum, visualize):
    x_min, y_min, x_max, y_max = roi
    small_frame_dim = 64
    G_overlays = {}
    for trck_class in tracked_classes:
        G_overlays[trck_class] = torch.zeros((small_frame_dim, small_frame_dim))

    for t in range(len(targets)):
        class_type = targets[t][0].item()

        center_x, center_y = targets[t][1] - [roi[0], roi[1]]
        w, h = targets[t][2]
        # if class_type == 5:
        #     tqdm.write("BUSS FOUND I REPEAT BUSS FOUND!!!!!!!!!!!")
        #     tqdm.write(f"x: {center_x}")
        #     tqdm.write(f"y: {center_y}")
        #     tqdm.write(f"w: {w}")
        #     tqdm.write(f"h: {h}")

        x_min, y_min = np.maximum(
            [0, 0], np.minimum([250, 250], np.array([center_x, center_y]) - [w / 2, h / 2])
        )

        x_max, y_max = np.maximum(
            [0, 0], np.minimum([250, 250], np.array([center_x, center_y]) + [w / 2, h / 2])
        )

        # print(x_min)
        # print(x_max)
        # Scale down the values to the 64x64 frame
        w_small = torch.tensor((x_max - x_min) * 64 / 250)
        h_small = torch.tensor((y_max - y_min) * 64 / 250)
        center_x_small = center_x * 64 / 250
        center_y_small = center_y * 64 / 250

        # Adjust sigma proportionally for the smaller frame
        sigma_x_small = w_small / 6  # * (1 - torch.exp(-(65 * abs(w_small / 64) + 0.1)))
        sigma_y_small = h_small / 6  # * (1 - torch.exp(-(65 * abs(h_small / 64) + 0.1)))
        events_factor = count_events(
            frame[int(y_min) : int(y_max), int(x_min) : int(x_max)],
            (y_min, y_max, x_min, x_max),
        )

        # if class_type == 5:
        #     center_x = torch.tensor(32)
        #     center_y = torch.tensor(32)
        #     sigma_x_small = torch.tensor(10)
        #     sigma_y_small = torch.tensor(10)

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

        G_overlays[class_type] = np.maximum(G_overlays[class_type], G_normalized)

        # Apply to the small overlay

    # if torch.max(G_overlays[class_type]) > clip_maximum:
    #     clip_maximum = torch.max(G_overlays[class_type])

    for trck_class in tracked_classes:
        overlays[trck_class] = leaky_integrator(
            G_overlays[trck_class].detach().clone(),
            tau=10,
            v=overlays[trck_class].detach().clone(),
        )
    x_min, y_min, x_max, y_max = roi
    if visualize:
        scaledLabels = []
        for trck_class in tracked_classes:
            temp = overlays[trck_class].detach().clone().numpy()
            temp = (temp / max(np.max(temp), 0.00001)) * 255

            scaledLabels.append(
                cv2.resize(
                    temp,
                    dsize=(250, 250),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            cv2.putText(
                scaledLabels[len(scaledLabels) - 1],
                f"class: {trck_class}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_4,
            )
        tempframe = np.concatenate(
            (
                frame.detach().clone() * 255,
                np.ones((250, 20)) * 255,
            ),
            axis=1,
        )
        for i, scaledLabel in enumerate(scaledLabels):
            if i == len(scaledLabels) - 1:
                tempframe = np.concatenate((tempframe, scaledLabel), axis=1)
            else:
                tempframe = np.concatenate(
                    (tempframe, scaledLabel, np.ones((250, 20)) * 255), axis=1
                )
        out.write(tempframe.astype(np.uint8))

    return frame.detach().clone(), overlays


H = transform.SimilarityTransform()
H.params = np.array(
    [
        [8.50065481e-01, 7.29692005e-03, 1.22287204e02],
        [-7.29692005e-03, 8.50065481e-01, 6.95716137e01],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# x_min, y_min, x_max, y_max
rois = [(90, 150, 340, 400), (575 - 250, 150, 575, 400)]  # []


def process_roi(i, j, clip_idx, roi, start_clip, end_clip, input_dir, save_dir, visualize):
    print(
        f"Processing clip: {i-start_clip+1}/{end_clip-start_clip+1} | ROI: {roi} ({j+1}/{len(rois)})"
    )
    targets = get_targets(f"{input_dir}track/labels/", 5400, i)
    frames_tensor = []
    labels_tensor = {}
    for trck_class in tracked_classes:
        labels_tensor[trck_class] = []
    data = torch.load(f"{input_dir}events/event_frames_{i}.pt")
    small_frame_dim = 64
    clip_maximum = 0
    overlays = {}
    for trck_class in tracked_classes:
        overlays[trck_class] = torch.zeros((small_frame_dim, small_frame_dim))
    out = None
    if visualize:
        out = cv2.VideoWriter(
            filename=f"{save_dir}{clip_idx+j}-vis.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=90,
            frameSize=(
                250 * (len(tracked_classes) + 1) + 20 * len(tracked_classes),
                250,
            ),
            isColor=False,
        )

    for frame_idx in tqdm(
        range(int(len(data))),
        desc=f"annotating frames",
        ncols=86,
        mininterval=0.25,
    ):
        frame = data[frame_idx].to_dense()[roi[1] : roi[3], roi[0] : roi[2]]
        warped = []

        if targets[frame_idx] == None:
            print("missing:", frame_idx)
            res = frame.detach().clone()

        elif len(targets[frame_idx]) == 0:
            res = frame.detach().clone()

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
                    [
                        targets[frame_idx][tar][0],
                        transformed_coordinates,
                        torch.tensor([w, h]),
                    ]
                )

            res, overlays = annotate_frame(
                frame, warped, overlays, out, roi, clip_maximum, visualize
            )

        frames_tensor.append(res)
        for trck_class in tracked_classes:
            labels_tensor[trck_class].append(overlays[trck_class].detach().clone())

    print(f"\nSaving visualization to \x1b[1m{save_dir}{clip_idx+j}-vis.mp4\x1b[22m")
    if visualize:
        out.release()
    print(f"\nSaving data to \x1b[1m{save_dir}{clip_idx+j}.pt\x1b[22m")
    clip_data = [torch.stack(frames_tensor).to_sparse()]

    for trck_class in tracked_classes:
        clip_data.append(torch.stack(labels_tensor[trck_class]).to_sparse())

    torch.save(
        clip_data,
        f"{save_dir}{clip_idx+j}.pt",
    )


def generate_labels(input_dir, save_dir, start_clip, end_clip, visualize):
    clip_idx = 103
    threads = [None, None]
    for i in range(start_clip, end_clip + 1):
        for j, roi in enumerate(rois):
            process_roi(i, j, clip_idx, roi, start_clip, end_clip, input_dir, save_dir, visualize)
        #     threads[j] = Thread(
        #         target=process_roi,
        #         args=[i, j, clip_idx, roi, start_clip, end_clip, input_dir, save_dir, visualize],
        #     )
        #     threads[j].start()
        # for j in range(len(threads)):
        #     threads[j].join()
        clip_idx += 2


generate_labels("w35/box1/1-09-04/", "clips/frames_with_labels/", 4, 6, True)

# w38/box3/3-09-27/
# w31/box2/2-08-01/
