import os
from threading import Thread
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import patches
from skimage import color, exposure, feature, filters, transform, util


visualize = False

img = 0


def process_clip(event_path, normal_path):
    global img
    torch.cuda.set_device(device=0)
    model_matches = 0
    total_error = 0
    event_data = torch.load(event_path)

    normal_cap = cv2.VideoCapture(normal_path)

    matches = []
    for i in tqdm(
        range(0, 5000, 50),
        ncols=90,
        mininterval=0.75,
    ):
        normal_cap.set(1, 0 + i)
        img_left2 = event_data[i + 1].to_dense()  # Read the Event frame
        ret, img_right = normal_cap.read()  # Read the Normal frame
        ret, img_right2 = normal_cap.read()  # Read the Normal frame

        img_left = exposure.adjust_gamma(filters.gaussian(img_left2, 10), 5)

        img_right = exposure.adjust_gamma(
            filters.gaussian(
                util.compare_images(
                    color.rgb2gray(img_right), color.rgb2gray(img_right2), method="diff"
                ),
                10,
            ),
            5,
        )

        img_left = exposure.rescale_intensity(img_left)
        img_right = exposure.rescale_intensity(img_right)

        blobs1 = feature.blob_doh(
            img_left, min_sigma=12, max_sigma=32, num_sigma=3, threshold_rel=0.45
        )
        blobs2 = feature.blob_doh(
            img_right, min_sigma=12, max_sigma=32, num_sigma=3, threshold_rel=0.55
        )

        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))

            axes[0].imshow(img_left, cmap="magma")
            axes[1].imshow(img_right, cmap="magma")

            for y1, x1, r1 in blobs1:
                cirk1 = patches.Circle((x1, y1), radius=r1, linewidth=2, edgecolor="r", fill=False)

                axes[0].add_patch(cirk1)

            for y2, x2, r2 in blobs2:

                cirk2 = patches.Circle((x2, y2), radius=r2, linewidth=2, edgecolor="r", fill=False)

                axes[1].add_patch(cirk2)

        for y1, x1, r1 in blobs1:
            best_match = 0.2
            best_index = -1
            for i, (y2, x2, r2) in enumerate(blobs2):
                error = math.dist((x1 / 640, y1 / 480), (x2 / 736, y2 / 460))
                if error < best_match:
                    best_match = error
                    best_index = i

            if best_index != -1:
                model_matches += 1
                total_error += best_match
                matches.append([[x1, y1], [x2, y2]])
                if visualize:
                    conn = patches.ConnectionPatch(
                        xyA=(x1, y1),
                        xyB=(blobs2[best_index][1], blobs2[best_index][0]),
                        coordsA="data",
                        coordsB="data",
                        axesA=axes[0],
                        axesB=axes[1],
                        color="lightgreen",
                        linewidth=3,
                    )

                    fig.add_artist(conn)

                    cirk1 = patches.Circle((x1, y1), radius=2, color="lightgreen", fill=True)
                    cirk2 = patches.Circle(
                        (blobs2[best_index][1], blobs2[best_index][0]),
                        radius=2,
                        color="lightgreen",
                        fill=True,
                    )

                    axes[0].add_patch(cirk1)
                    axes[1].add_patch(cirk2)

        if visualize:
            axes[0].axis("off")
            axes[1].axis("off")
            fig.tight_layout()
            plt.show()
            # fig.savefig(f"clips/test/{img}.png", bbox_inches="tight", pad_inches=0.1)
            # img += 1

            # plt.close(fig)

    normal_cap.release()

    matches = np.array(matches)
    model = transform.SimilarityTransform()
    weight = 0
    if len(matches) > 4:
        model.estimate(matches[:, 0], matches[:, 1])
        weight = 0.3 * model_matches + 0.7 * (model_matches - total_error * 10)

    return model, weight


def calc_matrix(event_folder, normal_folder, start_clip, end_clip, results):

    models = []
    weights = []
    for i in range(start_clip, end_clip + 1):
        model, weight = process_clip(
            f"{event_folder}event_frames_{i}.pt", f"{normal_folder}_{i}.mp4"
        )
        models.append(model.params)
        weights.append(weight)

    models = np.array(models)
    weights = np.array(weights)

    weights = weights / weights.sum()

    model = transform.SimilarityTransform()
    model.params = np.average(
        models,
        axis=0,
        weights=weights,
    )

    results.append(model.params)


models = []

if not visualize:
    t = Thread(
        target=calc_matrix,
        args=["../w31/box2/2-07-31/events/", "../w31/box2/2-07-31/normal/", 9, 11, models],
    )
    t.start()

    t1 = Thread(
        target=calc_matrix,
        args=["../w31/box2/2-07-31/events/", "../w31/box2/2-07-31/normal/", 7, 9, models],
    )
    t1.start()


calc_matrix("../w31/box2/2-07-31/events/", "../w31/box2/2-07-31/normal/", 4, 6, models)

if not visualize:
    t.join()
    t1.join()


model = transform.SimilarityTransform()


model.params = np.average(
    models,
    axis=0,
    weights=[1 / len(models)] * len(models),
)

print(repr(model))

# model = transform.SimilarityTransform()
# model.params = np.array(
#     [
#         [8.50065481e-01, 7.29692005e-03, 1.22287204e02],
#         [-7.29692005e-03, 8.50065481e-01, 6.95716137e01],
#         [0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )

event_data = torch.load("../w31/box2/2-07-31/events/event_frames_15.pt")

normal_cap = cv2.VideoCapture("../w31/box2/2-07-31/normal/_15.mp4")

normal_cap.set(1, 4)
img_left = event_data[4].to_dense()
ret, img_right = normal_cap.read()

target_tensor = None

with open(os.path.join("../w31/box2/2-07-31/labels/track/labels", "_15_4.txt")) as file:
    target_data = [
        float(value)
        for line in file
        if line.strip().split()[0] in ("2", "5", "7", "61")
        for value in line.split()[1:5]
    ]
    target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

axes[0].imshow(img_left, cmap="gray")
axes[1].imshow(img_right, cmap="brg")


for t in range(len(target_tensor)):
    # ===========================
    # Event Image
    # ===========================
    center_x, center_y = target_tensor[t][:2].numpy() * [736, 460]
    w, h = target_tensor[t][2:].numpy() * [736, 460]

    x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
    x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]

    x_min, y_min = model._apply_mat((x_min, y_min), model.inverse.params)[0]
    x_max, y_max = model._apply_mat((x_max, y_max), model.inverse.params)[0]

    center_x, center_y = model._apply_mat((center_x, center_y), model.inverse.params)[0]

    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor="r", facecolor="none"
    )
    cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
    axes[0].add_patch(rect)
    axes[0].add_patch(cirk)

    # ===========================
    # Normal Image
    # ===========================
    center_x, center_y = target_tensor[t][:2].numpy() * [736, 460]
    w, h = target_tensor[t][2:].numpy() * [736, 460]

    x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
    x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]

    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
    cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
    axes[1].add_patch(rect)
    axes[1].add_patch(cirk)


plt.tight_layout()
plt.show()


normal_cap.release()
