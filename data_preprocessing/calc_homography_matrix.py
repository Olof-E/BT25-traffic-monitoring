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


import math
import os
from pydoc import allmethods
from matplotlib import patches
import numpy as np
import cv2
import skimage
from skimage.color import rgb2gray
import skimage.exposure
from skimage.feature import match_descriptors, plot_matched_features
from skimage.measure import ransac
import matplotlib.pyplot as plt
from skimage.filters import rank
from skimage.morphology import disk, ball
from skimage.util import compare_images
from skimage.restoration import denoise_tv_chambolle


from skimage.feature import (
    match_descriptors,
    corner_peaks,
    corner_subpix,
    plot_matched_features,
    BRIEF,
)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

visualize = True
img = 0


def process_clip(event_path, normal_path):
    global img
    torch.cuda.set_device(device=0)
    model_matches = 0
    total_error = 0
    event_data = torch.load(event_path)

    normal_cap = cv2.VideoCapture(normal_path)

    matches = []
    model = None

    extractor = BRIEF()

    for i in tqdm(
        range(0, 5000, 150),
        ncols=86,
        mininterval=0.75,
        leave=False,
    ):
        for j in range(1):
            normal_cap.set(1, i + 10 * j)
            img_left2 = (
                event_data[i + 10 * j + 1].to_dense() + event_data[i + 10 * j + 2].to_dense()
            )  # Read the Event frame
            _, img_right = normal_cap.read()  # Read the Normal frame
            _ = normal_cap.grab()  # Read the Normal frame
            _ = normal_cap.grab()  # Read the Normal frame
            _ = normal_cap.grab()  # Read the Normal frame

            _, img_right2 = normal_cap.read()  # Read the Normal frame

            img_left = filters.gaussian(img_left2, 0.8)

            img_right = filters.gaussian(
                util.compare_images(
                    color.rgb2gray(img_right), color.rgb2gray(img_right2), method="diff"
                ),
                0.6,
            )

            img_left = exposure.rescale_intensity(img_left)
            img_right = exposure.rescale_intensity(img_right)

            keypoints1 = corner_peaks(img_left, min_distance=2, threshold_rel=0.1)
            keypoints2 = corner_peaks(img_right, min_distance=2, threshold_rel=0.1)

            extractor.extract(img_left, keypoints1)
            keypoints1 = keypoints1[extractor.mask]
            descriptors1 = extractor.descriptors

            extractor.extract(img_right, keypoints2)
            keypoints2 = keypoints2[extractor.mask]
            descriptors2 = extractor.descriptors

            try:
                matches12 = match_descriptors(
                    descriptors1, descriptors2, cross_check=True, max_distance=2, max_ratio=0.8
                )

                # fig, ax = plt.subplots(nrows=1, ncols=1)

                # plot_matched_features(
                #     img_left,
                #     img_right,
                #     keypoints0=keypoints1,
                #     keypoints1=keypoints2,
                #     matches=matches12,
                #     ax=ax,
                #     only_matches=True,
                # )
                # plt.show()

                # tqdm.write(f"matches found {matches12.shape[0]}")

                new_model, inliers = ransac(
                    (keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]]),
                    skimage.transform.SimilarityTransform,
                    min_samples=8,
                    residual_threshold=0.6,
                    max_trials=1500,
                )

                if inliers is not None and inliers.sum() >= 2:

                    fig, ax = plt.subplots(nrows=1, ncols=1)

                    plot_matched_features(
                        img_left,
                        img_right,
                        keypoints0=keypoints1,
                        keypoints1=keypoints2,
                        matches=matches12[inliers],
                        ax=ax,
                        only_matches=True,
                    )
                    plt.show()

                    if model is None:
                        # print(repr(new_model))
                        model = new_model
                        normal_cap.release()
                        return model, 1, 0.0000001

                        # model.estimate(keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]])
                    else:
                        # new_model = transform.SimilarityTransform()
                        # new_model.estimate(keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]])

                        model.params = np.average(
                            (model.params, new_model.params), weights=[0.95, 0.05], axis=0
                        )
                        # print(repr(model))

                # for match in matches12[inliers][0]:
                #     matches.append([keypoints1[match[0]], keypoints2[match[1]]])

            except:
                continue

    normal_cap.release()

    # matches = np.array(matches)
    # model = transform.SimilarityTransform()
    # weight = 0.000001
    # if len(matches) > 1:

    #     model.estimate(matches[:, 0], matches[:, 1])
    #     print(model.params)
    #     weight = 0 * model_matches + 1 * (1 - total_error)

    return model, 1, 0.0000001


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
                if line.strip().split()[0] in str(("0", "2", "5", "7"))
                for value in line.split()[0:5]
            ]
            target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 5)
            target_tensors[int(filename[2].rsplit(".", maxsplit=1)[0]) - 1] = target_tensor
    return target_tensors


def calc_matrix(event_folder, normal_folder, start_clip, end_clip, results, res_weights):

    models = []
    weights = []
    for i in range(start_clip, end_clip + 1):
        model, weight, total_error = process_clip(
            f"{event_folder}event_frames_{i}.pt", f"{normal_folder}_{i}.mp4"
        )
        if model is not None:
            models.append(model.params)
            weights.append(weight)

    models = np.array(models)
    weights = np.array(weights)

    weights = weights / weights.sum()

    model = transform.SimilarityTransform()
    model.params = np.average(
        models,
        axis=0,
    )
    print(repr(model))

    results.append(model.params)
    res_weights.append(1 - total_error)


mod_weights = []
models = []

curr_dir = "w31/box2/2-08-01"  # "w35/box1/1-09-04"

# "w31/box2/2-07-31"
# "w35/box1/1-09-04"

if not visualize:

    t1 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 22, 25, models, mod_weights],
    )
    t1.start()

    t2 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 16, 19, models, mod_weights],
    )
    t2.start()

    t3 = Thread(
        target=calc_matrix,
        args=[f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 10, 14, models, mod_weights],
    )
    t3.start()

calc_matrix(f"../{curr_dir}/events/", f"../{curr_dir}/normal/", 5, 9, models, mod_weights)

if not visualize:
    t1.join()
    t2.join()
    t3.join()


model = transform.SimilarityTransform()


mod_weights = np.array(mod_weights)
mod_weights = mod_weights / mod_weights.sum()

model.params = np.average(models, axis=0)

# model.params = model.inverse.params

print(repr(model))

# model = transform.SimilarityTransform()
# model.params = np.array(
#     [
#         [1.03138823e00, 5.43905205e-03, 6.67418124e00],
#         [-5.43905205e-03, 1.03138823e00, 3.32507537e01],
#         [0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )


event_data = torch.load(f"../{curr_dir}/events/event_frames_6.pt")

normal_cap = cv2.VideoCapture(f"../{curr_dir}/normal/_6.mp4")


target_tensors = get_targets(f"../{curr_dir}/track/labels/", 5400, 6)

# fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# axes[0].imshow(img_left, cmap="gray")
# axes[1].imshow(img_right, cmap="brg")

out = cv2.VideoWriter(
    filename=f"homography-vis3.mp4",
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=90,
    frameSize=(
        640 + 736,
        480,
    ),
    isColor=False,
)


for frame_idx in tqdm(
    range(int(len(event_data)) // 3),
    desc=f"annotating frames",
    ncols=86,
    mininterval=0.25,
):
    frame = event_data[frame_idx].to_dense() * 255
    ret, img_right = normal_cap.read()
    combined_frame = np.concatenate(
        (
            frame,
            np.concatenate((cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY), np.full((20, 736), 255))),
        ),
        axis=1,
    )
    for t in range(len(target_tensors[frame_idx])):
        # ===========================
        # Event Image
        # ===========================

        center_x, center_y = target_tensors[frame_idx][t][1:3].numpy() * [736, 460]
        w, h = target_tensors[frame_idx][t][3:].numpy() * [736, 460]

        x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
        x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]

        y_min, x_min = model._apply_mat((y_min, x_min), model.inverse.params)[0]
        y_max, x_max = model._apply_mat((y_max, x_max), model.inverse.params)[0]

        center_y, center_x = model._apply_mat((center_y, center_x), model.inverse.params)[0]

        cv2.rectangle(
            img=combined_frame,
            pt1=(int(x_min), int(y_min)),
            pt2=(int(x_max), int(y_max)),
            color=(255, 0, 0),
            thickness=2,
        )
        # rect = patches.Rectangle(
        #     (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor="r", facecolor="none"
        # )
        # cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
        # axes[0].add_patch(rect)
        # axes[0].add_patch(cirk)

        # ===========================
        # Normal Image
        # ===========================
        center_x, center_y = target_tensors[frame_idx][t][1:3].numpy() * [736, 460]
        w, h = target_tensors[frame_idx][t][3:].numpy() * [736, 460]

        x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
        x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]
        cv2.rectangle(
            img=combined_frame,
            pt1=(int(x_min + 640), int(y_min)),
            pt2=(int(x_max + 640), int(y_max)),
            color=(255, 0, 0),
            thickness=2,
        )

        # rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor="r", facecolor="none")
        # cirk = patches.Circle((center_x, center_y), radius=1, edgecolor="r", facecolor="red")
        # axes[1].add_patch(rect)
        # axes[1].add_patch(cirk)
    out.write(np.uint8(combined_frame))


# plt.tight_layout()
# plt.show()

out.release()
normal_cap.release()
