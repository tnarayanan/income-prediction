import torch
from income_pred_model import IncomePredModel
from PIL import Image
from torchvision import transforms

device = torch.device('cpu')

loaded_model = IncomePredModel(device)
state_dict = torch.load('model_params.pt', map_location=device)
loaded_model.conv_net.load_state_dict(state_dict)

def predict(img_path):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img = img.to(device=loaded_model.device)
    img = torch.unsqueeze(img, 0)

    prediction = loaded_model.conv_net(img)
    prediction = torch.squeeze(prediction)
    prediction = prediction.item()
    return prediction
