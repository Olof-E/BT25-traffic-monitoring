import os
import numpy as np
import matplotlib.pyplot as plt

# Define the class names
class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def count_classes(label_dir):
    # Initialize a NumPy array to hold the counts for each class
    class_counts = np.zeros(8, dtype=int)

    # Iterate over all files in the label directory
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    if class_id in class_names:
                        class_counts[class_id] += 1

    return class_counts

def plot_class_counts(class_counts):
    classes = list(class_names.keys())
    counts = [class_counts[cls] for cls in classes]
    class_labels = [class_names[cls] for cls in classes]

    plt.figure(figsize=(12, 8))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Detections')
    plt.title('Number of Detections per Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

label_dir = "/home/tobias/Documents/videos/2024-04-18/box2/yolo/results/track/labels"  # Replace with the label directory
class_counts = count_classes(label_dir)
plot_class_counts(class_counts)