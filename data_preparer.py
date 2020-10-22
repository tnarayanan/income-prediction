from data_input import DataInput
import torch
import os


class DataPreparer(object):

    def __init__(self, batch_size=64):
        self.train_dataset, self.test_dataset = self.upload_data()
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

    def upload_data(self):
        if os.path.exists("./data/test_dataset.pt") and os.path.exists("./data/train_dataset.pt"):
            return torch.load('./data/train_dataset.pt'), torch.load('./data/test_dataset.pt')

        train_data = DataInput()
        test_data = DataInput(test_data=True)

        train_data.load_data()
        test_data.load_data()

        train_dataset = torch.utils.data.TensorDataset(train_data.x, train_data.Y)
        test_dataset = torch.utils.data.TensorDataset(test_data.x, test_data.Y)

        torch.save(train_dataset, 'data/train_dataset.pt')
        torch.save(test_dataset, 'data/test_dataset.pt')

        return train_dataset, test_dataset
