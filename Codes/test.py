import numpy as np
import os
import glob
from PIL import Image, ImageDraw
from DataLoader import Dataset
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils import data
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('Image', type=str, help='provide file path to the image, e.g.: c:/home/data/train/120.jpg')
args = parser.parse_args()

TIME_STAMP = '2020-01-27-15-39-28'#'2020-01-27-15-34-02'
project_dir = r'/home/banikr/PycharmProjects/Project/PhoneCo'

testImage = glob.glob(os.path.join(args.Image))
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = models.resnet18(pretrained=False).to(device)
num_ftrs = model.fc.in_features
# print(num_ftrs)
# print(model_ft,model)
model.fc = nn.Linear(num_ftrs, 2)
# model = nn.Sequential(model, nn.Linear(2,2))
model = model.to(device)
# print(model)

FILEPATH_MODEL_LOAD = os.path.join(project_dir,'{}.pt'.format(TIME_STAMP))
train_states = torch.load(FILEPATH_MODEL_LOAD)
# print(train_states.keys())
model.load_state_dict(train_states['train_states_best']['model_state_dict'])

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])],)
Input = Image.open(testImage[0]).convert('RGB')
Input = transform(Input)
Input = Input.float().to(device)
Input = Input.unsqueeze(0)
output = torch.sigmoid(model(Input))
output = output.detach().cpu().numpy()
print('{0:.4f}'.format(output[0][0]))
print('{0:.4f}'.format(output[0][1]))

# print(train_states['train_states_best']['epoch'])

x = 490
y = 326
xrad = np.ceil(5)
yrad = np.ceil(5)
Xcor = round(x*output[0][0])
Ycor = round(y*output[0][1])

Im = Image.open(testImage[0])
draw = ImageDraw.Draw(Im)
draw.ellipse((Xcor-xrad,Ycor-yrad,Xcor+xrad,Ycor+yrad), fill = (255,0,0), outline=(255,0,0))

plt.imshow(Im)
plt.show()
