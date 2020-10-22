from conv_net import ConvNet
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def mean_squared_error(expected, output):
    return torch.mean(torch.pow(output - expected, 2)).item()


class IncomePredModel(object):

    def __init__(self, device, lr=0.001):
        self.device = device
        self.conv_net = ConvNet().to(device=self.device)
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.conv_net.parameters(), lr=lr)

    def train(self, data, num_epochs, plot_loss=True):
        loss_train = []

        print("Training model...")
        total = num_epochs * len(data.train_dataset)
        with tqdm(total=total) as pbar:
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (x, Y) in enumerate(data.train_loader):
                    x = x.to(device=self.device)
                    Y = Y.to(device=self.device)

                    self.optimizer.zero_grad()  # resets the information from last time
                    pred_y = self.conv_net(x)  # calculates the predictions
                    pred_y = torch.reshape(pred_y, (len(Y),))

                    loss = self.criterion(pred_y, Y)  # calculates the loss
                    loss.backward()  # gradient descent, part 1
                    self.optimizer.step()  # gradient descent, part 2

                    pbar.update(len(x))

                    epoch_loss += loss.item() / len(Y)
                # print(f"Epoch {epoch}: {epoch_loss}")
                loss_train.append(epoch_loss)
        print("Training finished.")
        if plot_loss:
            plt.plot(loss_train, label='Training loss')
            plt.legend()
            plt.show()

    def evaluate(self, data, on='test'):
        expecteds = []
        outputs = []
        dataset = None
        if on == 'test':
            dataset = data.test_loader
        else:
            dataset = data.train_loader
        with torch.no_grad():
            for batch_idx, (x, Y) in enumerate(dataset):
                x = x.to(device=self.device)
                Y = Y.to(device=self.device)
                output = self.conv_net(x)
                expected = Y
                output = torch.reshape(output, (len(output),))

                outputs.append(output)
                expecteds.append(expected)

        outputs = torch.cat(outputs)
        expecteds = torch.cat(expecteds)
        #  find mean squared error
        mse = mean_squared_error(expecteds, outputs)
        print("Mean Squared Error on", on, "set:", mse)
