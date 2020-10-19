#Clare
from data_input import DataInput
from conv_net import ConvNet
from tqdm import tqdm, trange
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data = DataInput()
test_data = DataInput(test_data=True)

train_data.load_data()
test_data.load_data()

train_dataset = torch.utils.data.TensorDataset(train_data.x, train_data.Y)
test_dataset = torch.utils.data.TensorDataset(test_data.x, test_data.Y)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# split into mini-batches of 64

conv_net = ConvNet().to(device=device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(conv_net.parameters(), lr=0.001)

loss_train = []
num_epochs = 10

print("Training model...")
total = num_epochs*len(train_dataset)
with tqdm(total=total) as pbar:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (x, Y) in enumerate(train_loader):
            x = x.to(device=device)
            Y = Y.to(device=device)

            optimizer.zero_grad()  # resets the information from last time
            pred_y = conv_net(x)  # calculates the predictions
            pred_y = torch.reshape(pred_y, (len(Y),))

            loss = criterion(pred_y, Y)  # calculates the loss
            loss.backward()  # gradient descent, part 1
            optimizer.step()  # gradient descent, part 2

            pbar.update(len(x))

            epoch_loss += loss.item() / len(Y)
        # print(f"Epoch {epoch}: {epoch_loss}")
        loss_train.append(epoch_loss)

print("Training finished.")

plt.plot(loss_train, label='Training loss')
plt.legend()
plt.show()

# Calculate training accuracy
total_abs_perc_error = 0
total = len(train_dataset)
with torch.no_grad():
    for batch_idx, (x, Y) in enumerate(train_loader):
        x = x.to(device=device)
        Y = Y.to(device=device)
        output = conv_net(x)
        expected = Y
        output = torch.reshape(output, (len(output),))

        perc_error = abs(output - expected) / expected
        total_abs_perc_error += sum(perc_error).item()
        # if i % 100 == 0:
        #     print('expected', expected, 'predicted', output)
print(f"Average absolute percent error on training set: {total_abs_perc_error*100/total}%")

# Calculate test set accuracy
print()
print("Evaluating on test data...")
total_abs_perc_error = 0
total = len(test_dataset)
with torch.no_grad():
    # trange automatically creates a progress bar
    for x, Y in test_loader:
        x = x.to(device=device)
        Y = Y.to(device=device)
        output = conv_net(x)
        expected = Y
        output = torch.reshape(output, (len(output),))

        perc_error = abs(output - expected) / expected
        total_abs_perc_error += sum(perc_error).item()
        # if i % 100 == 0:
        #     print('expected', expected, 'predicted', output)


print(f"Average absolute percent error on test set: {total_abs_perc_error*100/total}%")

