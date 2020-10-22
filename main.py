# Zander's branch

from data_preparer import DataPreparer
from model import IncomePredModel
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data = DataPreparer()

model = IncomePredModel(device)

num_epochs = 3

model.train(data, num_epochs)

model.evaluate(data, on='test')
model.evaluate(data, on='train')
