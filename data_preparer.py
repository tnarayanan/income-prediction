from data_input import DataInput
import torch
import random
from custom_dataset import CustomDataset
import os


class DataPreparer():

    def __init__(self, batch_size=64, use_shortcut=True, transform=None):
        self.use_shortcut = use_shortcut
        self.transform = transform
        self.train_dataset, self.test_dataset = self.upload_data()
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

    def upload_data(self):
        data = DataInput()
        if self.use_shortcut and os.path.exists("./data/data_X.pt") and os.path.exists("./data/data_y.pt"):
            data.X = torch.load('./data/data_X.pt')
            data.y = torch.load('./data/data_y.pt')
        else:
            data.load_data()

            torch.save(data.X, 'data/data_X.pt')
            torch.save(data.y, 'data/data_y.pt')

        inds = range(len(data.X))
        test_inds = random.sample(inds, int(0.2 * len(data.X)))
        train_inds = [i for i in inds if i not in test_inds]
        train_X = data.X[train_inds, :, :, :]
        test_X = data.X[test_inds, :, :, :]

        train_y = data.y[train_inds]
        test_y = data.y[test_inds]

        train_dataset = CustomDataset(train_X, train_y, transform=self.transform)
        test_dataset = CustomDataset(test_X, test_y)

        return train_dataset, test_dataset
