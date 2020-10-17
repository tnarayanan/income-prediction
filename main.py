from data_input import DataInput
from conv_net import ConvNet
from tqdm import tqdm, trange
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data = DataInput()
train_data.load_data()

test_data = DataInput(test_data=True)
test_data.load_data()

# split into mini-batches of 64
train_batch_x = torch.split(train_data.x, 64)
train_batch_Y = torch.split(train_data.Y, 64)

conv_net = ConvNet().to(device=device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(conv_net.parameters(), lr=0.001)

loss_train = []
num_epochs = 10

print("Training model...")
total = num_epochs*len(train_data.x)
with tqdm(total=total) as pbar:
    for epoch in range(10):
        for i in range(len(train_batch_x)):
            optimizer.zero_grad()  # resets the information from last time
            pred_y = conv_net(train_batch_x[i].to(device=device))  # calculates the predictions
            pred_y = torch.reshape(pred_y, (len(train_batch_Y[i]),))

            loss = criterion(pred_y, train_batch_Y[i].to(device=device))  # calculates the loss
            loss.backward()  # gradient descent, part 1
            optimizer.step()  # gradient descent, part 2

            pbar.update(len(train_batch_x[i]))

print("Training finished.")

# Calculate training accuracy
total_abs_perc_error = 0
total = len(train_data.x)
with torch.no_grad():
    # trange automatically creates a progress bar
    for i in trange(len(train_data.x)):
        output = conv_net(train_data.x[i].reshape((1, 3, 256, 256)).to(device=device)).item()
        expected = train_data.Y[i].item()
        total_abs_perc_error += abs(output - expected) / expected

print(f"Average absolute percent error on training set: {total_abs_perc_error*100/total}%")

# Calculate test set accuracy
print()
print("Evaluating on test data...")
total_abs_perc_error = 0
total = len(test_data.x)
with torch.no_grad():
    # trange automatically creates a progress bar
    for i in trange(len(test_data.x)):
        output = conv_net(test_data.x[i].reshape((1, 3, 256, 256)).to(device=device)).item()
        expected = test_data.Y[i].item()
        total_abs_perc_error += abs(output - expected) / expected
        # if i % 100 == 0:
        #     print('expected', expected, 'predicted', output)


print(f"Average absolute percent error on test set: {total_abs_perc_error*100/total}%")
