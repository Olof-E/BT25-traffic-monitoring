import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import patches
from skimage import color, exposure, feature, filters, transform, util

cap = cv2.VideoCapture("clips/events/event_clip_4.mp4")  # video_name is the video being called
cap2 = cv2.VideoCapture("2-08/_4.mp4")  # video_name is the video being called


matches = []

visualize = True

for i in tqdm(
    range(0, 5000, 50),
    ncols=90,
    mininterval=0.75,
):
    cap.set(1, 1 + i)
    cap2.set(1, 0 + i)
    ret, img_left2 = cap.read()  # Read the Event frame
    ret, img_right = cap2.read()  # Read the Normal frame
    cap2.read()
    ret, img_right2 = cap2.read()  # Read the Normal frame

    img_left = exposure.adjust_gamma(filters.gaussian(color.rgb2gray(img_left2), 10), 2)
    # img_left = skimage.exposure.rescale_intensity(
    #     denoise_tv_chambolle(rgb2gray(img_left2), weight=2000, max_num_iter=100),
    #     out_range=(0, 255),
    # )

    img_right = exposure.adjust_gamma(
        filters.gaussian(
            util.compare_images(
                color.rgb2gray(img_right), color.rgb2gray(img_right2), method="diff"
            ),
            10,
        ),
        2,
    )

    img_left = exposure.rescale_intensity(img_left)
    img_right = exposure.rescale_intensity(img_right)

    blobs1 = feature.blob_doh(img_left, min_sigma=18, max_sigma=32, num_sigma=3, threshold_rel=0.45)
    blobs2 = feature.blob_doh(
        img_right, min_sigma=18, max_sigma=32, num_sigma=3, threshold_rel=0.55
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
        best_match = 0.15
        best_index = -1
        for i, (y2, x2, r2) in enumerate(blobs2):
            error = math.dist((x1 / 640, y1 / 480), (x2 / 736, y2 / 460))
            if error < best_match:
                best_match = error
                best_index = i

        if best_index != -1:
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
        plt.tight_layout()
        plt.show()


matches = np.array(matches)
model = transform.SimilarityTransform()
model.estimate(matches[:, 0], matches[:, 1])

print(repr(model.params))
print(repr(model.inverse.params))


cap.release()
cap2.release()

cap = cv2.VideoCapture("clips/events/event_clip_2.mp4")  # video_name is the video being called
cap2 = cv2.VideoCapture("2-08/_2.mp4")  # video_name is the video being called


cap.set(1, 3690)
cap2.set(1, 3690)
ret, img_left = cap.read()  # Read the frame
ret, img_right = cap2.read()  # Read the frame

target_tensor = None

with open(os.path.join("./yolo/results/track2/labels", "_2_3690.txt")) as file:
    target_data = [
        float(value)
        for line in file
        if line.strip().split()[0] in ("2", "5", "7", "61")
        for value in line.split()[1:5]
    ]
    target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 4)
    print(repr(target_tensor))

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

axes[0].imshow(img_left, cmap="brg")
axes[1].imshow(img_right, cmap="brg")


for t in range(len(target_tensor)):
    center_x, center_y = target_tensor[t][0] * 736, target_tensor[t][1] * 460
    w, h = target_tensor[t][2] * 640, target_tensor[t][3] * 480

    a = model._apply_mat((center_x, center_y), model.inverse.params)[0]

    center_x = a[0]  # * (640 / 736)
    center_y = a[1]  # * (480 / 460)

    x_min = center_x - w / 2
    y_min = center_y - h / 2
    x_max = center_x + w / 2
    y_max = center_y + h / 2

    # Create and add rectangle patch and mark circle for bbox
    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
    cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
    axes[0].add_patch(rect)
    axes[0].add_patch(cirk)

for t in range(len(target_tensor)):
    center_x, center_y = target_tensor[t][0] * 736, target_tensor[t][1] * 460
    w, h = target_tensor[t][2] * 736, target_tensor[t][3] * 460

    # id = targets[t][4]
    # print(id)
    x_min = center_x - w / 2
    y_min = center_y - h / 2
    x_max = center_x + w / 2
    y_max = center_y + h / 2

    # Create and add rectangle patch and mark circle for bbox
    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
    cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
    axes[1].add_patch(rect)
    axes[1].add_patch(cirk)

plt.tight_layout()
plt.show()


cap.release()
cap2.release()
