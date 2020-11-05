from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        X = self.tensors[0][index]

        if self.transform:
            X = self.transform(X)

        y = self.tensors[1][index]

        return X, y

    def __len__(self):
        return self.tensors[0].size(0)
