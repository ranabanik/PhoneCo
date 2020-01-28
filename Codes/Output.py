import numpy as np
import os
import glob
from PIL import Image
from DataLoader import Dataset
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils import data
#'2020-01-27-15-34-02' #'2020-01-27-21-16-33' #
TIME_STAMP = '2020-01-27-15-39-28' #'2020-01-27-15-34-02' #'2020-01-27-21-16-33' #'2020-01-27-15-34-02'##'2020-01-27-03-07-53'#'2020-01-26-22-17-46'
valid_dir = r'/media/banikr/DATA/CS8395assignment1_data/validation'
test_dir = r'/media/banikr/DATA/CS8395assignment1_data/test'
project_dir = r'/home/banikr/PycharmProjects/Project/PhoneCo'

valImages = glob.glob(os.path.join(valid_dir, '*.jpg'))
testImages = glob.glob(os.path.join(test_dir, '*.jpg'))
# print(testImages[6])
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

# valid_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# input = torch.rand([1,3,326,490]).to(device)
# output = model(input)
# print(output.shape)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])],)
# val_dataset = Dataset(train_dir, trainImages, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

predict = []
for i in testImages:
    Input = Image.open(i).convert('RGB')
    # print(np.array(Input).max()) #255
    Input = transform(Input)
    # Input = transforms.ToTensor()(Input.unsqueeze(0))
    Input = Input.float().to(device)
    Input = Input.unsqueeze(0)
    # print(Input.shape,Input.max(), Input.min())
    output = torch.sigmoid(model(Input))
    # output = output.detach().cpu().numpy()
    # print(output.shape)
    print(output)
    predict.append([os.path.basename(i), output[0],])

print(predict)

print(train_states['train_states_best']['epoch'])