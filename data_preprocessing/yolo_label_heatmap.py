import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import cv2

def read_labels(label_dir, class_id, video_resolution):
    heatmap_data = []

    # Iterate over all files in the label directory
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as file:
                for line in file:
                    data = line.split()
                    if int(data[0]) == class_id:
                        x_center = float(data[1]) * video_resolution[0]
                        y_center = float(data[2]) * video_resolution[1]
                        heatmap_data.append((x_center, y_center))

    return heatmap_data

def create_heatmap(heatmap_data, output_file, video_resolution, video_frame):
    # Convert the list of coordinates to a NumPy array
    heatmap_data = np.array(heatmap_data)

    # Create a 2D histogram to use as the heatmap data
    heatmap, xedges, yedges = np.histogram2d(heatmap_data[:, 0], heatmap_data[:, 1], bins=50, range=[[0, video_resolution[0]], [0, video_resolution[1]]])

    # Normalize the heatmap
    heatmap = heatmap / np.max(heatmap)

    # Transpose the heatmap to correct the orientation
    heatmap = heatmap.T

    # Resize the heatmap to match the video frame size
    heatmap_resized = cv2.resize(heatmap, (video_resolution[0], video_resolution[1]))

    # Convert the heatmap to a color map
    heatmap_colormap = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Overlay the heatmap on the video frame
    overlay = cv2.addWeighted(video_frame, 0.3, heatmap_colormap, 0.9, 0)

    # Save and display the result
    cv2.imwrite(output_file, overlay)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap Overlay')
    plt.axis('off')
    plt.show()

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)

def get_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Error reading frame from video file {video_path}")
    cap.release()
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a heatmap from YOLO label files.")
    parser.add_argument("label_dir", help="Path to the directory containing YOLO label files.")
    parser.add_argument("class_id", type=int, help="Class ID to generate heatmap for.")
    parser.add_argument("output_file", help="Path to save the output heatmap image.")
    args = parser.parse_args()

    video_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.label_dir))))
    label_video_dir = os.path.dirname(args.label_dir)

    # Find the name of the video file used in the yolo track
    for label_video_name in os.listdir(label_video_dir):
        if label_video_name.endswith((".mp4", ".avi", ".mov")):
            break

    # Find the original video file without labels
    video_file = None
    for filename in os.listdir(video_dir):
        if filename == label_video_name:
            video_file = os.path.join(video_dir, filename)
            break

    if not video_file:
        raise ValueError("No video file found in the directory above the labels")

    video_resolution = get_video_resolution(video_file)
    video_frame = get_video_frame(video_file)
    heatmap_data = read_labels(args.label_dir, args.class_id, video_resolution)
    create_heatmap(heatmap_data, args.output_file, video_resolution, video_frame)