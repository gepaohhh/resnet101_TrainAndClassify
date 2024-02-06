import torchvision
import torchvision.models as models
import torch

device = torch.device('cuda')
network = models.resnet101().to(device=device)
path = r"D:\StreetPicture\picture_recognition\classify\log\last.pt"
load = network.load_state_dict(torch.load(path, map_location=device))
print(load)