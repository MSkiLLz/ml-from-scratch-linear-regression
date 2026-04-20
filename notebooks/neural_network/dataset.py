from torch.utils.data import Dataset, DataLoader
import torch

class SimpleDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([[1.0,2.0],
                               [2.0,3.0],
                               [3.0,4.0],
                               [5.0,6.0]])
        self.y = torch.tensor([[0.0],[0.0],[1.0],[1.0]])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)