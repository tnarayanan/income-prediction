import torch
from torchvision import transforms
from data_preparer import DataPreparer
from income_pred_model import IncomePredModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

modifications = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomChoice([
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((-90, -90)),
        transforms.RandomRotation((0, 0))
    ]),
    # transforms.RandomResizedCrop(256, scale=(0.7, 1)),
    transforms.ToTensor()
])

no_modifications = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

transform = transforms.RandomChoice([
    modifications,
    no_modifications
])

data = DataPreparer(use_shortcut=True, transform=transform)

model = IncomePredModel(device)

model.train(data, 1)

torch.save(model.conv_net.state_dict(), 'model_params.pt')

model.evaluate(data, on='test')
model.evaluate(data, on='train')
