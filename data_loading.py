import numpy as np
import torch
import tqdm as tqdm
from sklearn.model_selection import KFold, train_test_split
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

        return frame, target


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


def get_data(fpath):
    data = torch.load(fpath)
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
    )

    target_sequences = create_target_sequences(targets, sequence_length, overlap_length, batch_size)

    # kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    # for train_index, test_index in kfold.split(frames_sequence):

    #     X_train, X_test = frames_sequence[train_index], frames_sequence[test_index]
    #     y_train, y_test = target_sequences[train_index], target_sequences[test_index]

    X_train, X_test, y_train, y_test = train_test_split(
        frames_sequence,
        target_sequences,
        test_size=0.2,
        random_state=42,
    )

    dataset = EventDataset(X_train, y_train)
    test_dataset = EventDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=18,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=18,
    )

    return train_loader, test_loader
