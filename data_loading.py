import gc
import numpy as np
import torch
import tqdm as tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sequence_length, overlap_length, batch_size = 75, 25, 16


class EventDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        target = self.targets[idx]
        # target2 = self.targets[idx][1]
        # target3 = self.targets[idx][2]
        # target4 = self.targets[idx][3]

        return frame, target  # , target2, target3, target4


def create_frame_sequences(data, sequence_length, overlap_length, batch_size):
    sequences = []
    start_index = 0

    while start_index + sequence_length <= len(data):
        sequence = data[start_index : start_index + sequence_length]
        sequences.append(sequence)
        start_index += sequence_length - overlap_length

    num_sequences = (len(sequences) // batch_size) * batch_size
    sequences = sequences[:num_sequences]

    return sequences


def create_target_sequences(data, sequence_length, overlap_length, batch_size):
    sequences = []
    start_index = 0

    while start_index + sequence_length <= len(data[0]):
        sequence1 = data[0][start_index : start_index + sequence_length]
        sequence2 = data[1][start_index : start_index + sequence_length]
        sequence3 = data[2][start_index : start_index + sequence_length]
        sequence4 = data[3][start_index : start_index + sequence_length]

        sequences.append([sequence1, sequence2, sequence3, sequence4])
        start_index += sequence_length - overlap_length

    num_sequences = (len(sequences) // batch_size) * batch_size
    sequences = sequences[:num_sequences]

    return torch.from_numpy(np.asarray(sequences))


def get_data(number):
    # try:
    data = torch.load(f"./clips/frames_with_labels/{number}-0.pt")
    if not data[0].is_coalesced():
        data[0] = data[0].coalesce()

    for i in range(4):
        if not data[i + 1].is_coalesced():
            data[i + 1] = data[i + 1].coalesce()

    frames_tensor = data[0].to_dense().float()
    targets = []
    for i in range(4):
        targets.append(data[i + 1].to_dense().float())

    frames_sequence = create_frame_sequences(
        frames_tensor, sequence_length, overlap_length, batch_size
    )  # torch.Size([536, 50, 200, 200])

    target_sequences = create_target_sequences(targets, sequence_length, overlap_length, batch_size)

    # target_sequence = create_sequences(
    #     targets[0], sequence_length, overlap_length, batch_size
    # )  # torch.Size([536, 50, 64, 64])

    # for i in range(4):
    #     target_sequence = create_sequences(
    #         targets[i], sequence_length, overlap_length, batch_size
    #     )  # torch.Size([536, 50, 64, 64])
    #     for j in range(len(frames_sequence)):
    #         target_sequences[j][i] = np.append(target_sequences[j][i], target_sequence[j])
    # print(np.array(frames_sequence).shape)
    # print(np.array(target_sequences).shape)

    X_train, X_test, y_train, y_test = train_test_split(
        frames_sequence, target_sequences, test_size=0.2, random_state=42
    )

    dataset = EventDataset(X_train, y_train)
    test_dataset = EventDataset(X_test, y_test)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

    # print(len(frames_tensor))
    # print(len(frames_sequence))
    return train_loader, test_loader


# except Exception as e:
#     print(f"Error loading file {number}-0.pt: {e}")
#     return None
