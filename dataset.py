import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from load_data import tokenize_sequence

# Create a dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, annotations_file, max_length):
        self.df = pd.read_csv(annotations_file)
        self.sequences = self.df["sequence"]
        self.next_characters = self.df["next_character"]

        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        sequence = tokenize_sequence(self.sequences[item].lower())
        next_character = tokenize_sequence(self.next_characters[item].lower())

        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        elif len(sequence) < self.max_length:
            padding = torch.full((self.max_length - len(sequence)), -1)
            sequence = torch.cat((sequence, padding))
        return {
            "sequence" : torch.tensor(sequence, dtype=torch.long),
            "next_character" : torch.tensor(next_character, dtype=torch.long)
        }

def create_datasets(dataset):
    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    return train_data, test_data

def create_dataloaders(train_data, test_data):

    train_dataloader, test_dataloader = DataLoader(
        dataset=train_data,
        batch_size=16,
        shuffle=False
    ), DataLoader(
        dataset=test_data,
        batch_size=16,
        shuffle=False
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    dataset = ShakespeareDataset(annotations_file="annotations_file.csv", max_length=100)
    train_data, test_data = create_datasets(dataset)

    train_dataloader, test_dataloader = create_dataloaders(train_data=train_data, test_data=test_data)

    for batch in test_dataloader:
        print(batch["next_character"])
        break
