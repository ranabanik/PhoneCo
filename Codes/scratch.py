# import torchvision.models as models
# model_names = sorted(name for name in models.__dict__ if not name.startswith(
#     "__") and callable(models.__dict__[name]))
# print(model_names)
# arch = 'ResNet'
# print('==> Creating model {}...'.format(arch))
# model = models.__dict__[arch]()

from model import VGGBase, ResNet_PC, weights_init
import torch
import torch.nn as nn
import os
import glob
import time
from DataLoader import Dataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = r'C:\CS8395_DLMIC\data\assignment1_data'
train_dir = os.path.join(data_dir, 'train')
label_dir = r'C:\CS8395_DLMIC\data\assignment1_data\labels'
label_txt = glob.glob(os.path.join(label_dir, '*els.txt'))

model = ResNet_PC().to(device)
torch.cuda.manual_seed(1)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr= 0.005)
criterion = nn.MSELoss().cuda()

with open(os.path.join(label_txt[0]), 'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]

transform = transforms.ToTensor()

train_dataset = Dataset(train_dir, filenames, transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# model = ResNet_PC().to(device)
# model.apply(weights_init)
if __name__ !="__main__":
    input = torch.rand([4,3,326,490]).float().to(device)
    # print(input)
    # print(model)
    output = model(input)
    print("Output shape: ", output.shape) #Output shape:  torch.Size([4, 2])

for tBatchIdx, sample in enumerate(train_loader):
    print("training...")
    # time_batch_load = time.time() - time_batch_start
    Image = sample[0].float().to(device)
    # print(np.array(Image).shape)   # 4, 3, 326, 490
    # print('Data: ', torch.min(Image), torch.max(Image)) #0~1
    X = sample[1].float().to(device)
    Y = sample[2].float().to(device)
    output = model(Image)
    break

output = output.detach().cpu().numpy()
print(output)