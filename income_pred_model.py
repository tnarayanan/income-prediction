from conv_net import ConvNet
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class IncomePredModel(object):

    def __init__(self, device, lr=0.001):
        self.device = device
        self.conv_net = ConvNet().to(device=self.device)
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.conv_net.parameters(), lr=lr)

    def train(self, data, num_epochs, plot_loss=True):
        loss_train = []
        loss_test = []
        print("Training model...")
        total = num_epochs * len(data.train_dataset)
        with tqdm(total=total) as pbar:
            for epoch in range(num_epochs):
                epoch_loss_t, epoch_loss_v = (0, 0)
                for batch_idx, (x, Y) in enumerate(data.train_loader):
                    x = x.to(device=self.device)
                    Y = Y.to(device=self.device)

                    self.optimizer.zero_grad()  # resets the information from last time
                    pred_y = self.conv_net(x)  # calculates the predictions
                    pred_y = torch.reshape(pred_y, (len(Y),))

                    loss = self.criterion(pred_y, Y)  # calculates the loss
                    loss.backward()  # gradient descent, part 1
                    torch.nn.utils.clip_grad_norm_(self.conv_net.parameters(), 50)
                    self.optimizer.step()  # gradient descent, part 2

                    pbar.update(len(x))

                    epoch_loss_t += loss.item() / len(Y)

                for batch_idx, (x, Y) in enumerate(data.test_loader):
                    x = x.to(device=self.device)
                    Y = Y.to(device=self.device)
                    pred_y = torch.squeeze(self.conv_net(x))
                    loss = self.criterion(pred_y, Y)

                    epoch_loss_v += loss.item() / len(Y)

                epoch_loss_t /= len(data.train_loader)
                epoch_loss_v /= len(data.test_loader)

                print(f"Epoch {epoch}: train {epoch_loss_t}, test {epoch_loss_v}")

                loss_train.append(epoch_loss_t)
                loss_test.append(epoch_loss_v)
        print("Training finished.")
        if plot_loss:
            plt.plot(loss_train, label='Training loss')
            plt.plot(loss_test, label='Test Loss')
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
                output = torch.squeeze(self.conv_net(x))
                expected = Y

                outputs.append(output)
                expecteds.append(expected)

        outputs = torch.cat(outputs)
        expecteds = torch.cat(expecteds)
        l1 = torch.nn.L1Loss()(expecteds, outputs)
        print("L1 on", on, "set:", l1.item())
