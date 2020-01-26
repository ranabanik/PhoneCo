"""rana.banik@vanderbilt.edu"""
"""Image size (326, 490, 3). The labels include total 115 images. 105 images are in train folder and rest are in validation."""
"""For data augmentation consider rotation as the coordinates being center"""
import numpy as np
import os
import glob
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from DataLoader import Dataset
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR #to vary learning rate
from model import ResNet_PC, weights_init, Resnet_mob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TIME_STAMP  = time.strftime('%Y-%m-%d-%H-%M-%S')
print(TIME_STAMP)

label_dir = r'C:\CS8395_DLMIC\data\assignment1_data\labels'
train_dir = r'C:\CS8395_DLMIC\data\assignment1_data\train'
label_txt = glob.glob(os.path.join(label_dir, '*els.txt'))
# coOrdinates_txt = glob.glob(os.path.join(label_dir,'*tes.txt'))
# x_txt = glob.glob(os.path.join(label_dir, 'Xcoor.txt'))
# y_txt = glob.glob(os.path.join(label_dir, 'Ycoor.txt'))
dir_project = r'C:\Users\ranab\OneDrive\PycharmProjects\PhoneCo\Project'
dir_model = r'C:\Users\ranab\OneDrive\PycharmProjects\Models'
dir_log = os.path.join(dir_project,'log')
this_project_log = os.path.join(dir_log, TIME_STAMP)
os.mkdir(this_project_log)

data_dir = r'C:\CS8395_DLMIC\data\assignment1_data'
train_dir = os.path.join(data_dir, 'train')

FILEPATH_LOG = os.path.join(this_project_log, '{}.bin'.format(TIME_STAMP))

with open(os.path.join(label_txt[0]), 'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]
# # print(filenames)
# print(len(filenames))
# imagePath = os.path.join(train_dir,filenames[0].split()[0])
# # imagePath = glob.glob(os.path.join(train_dir,filenames[16]))
# print(imagePath)
# #
# #
# X = Image.open(imagePath).convert("RGB")
# print(np.array(X).shape)
#
# print(filenames[0].split()[2])
# X = filenames[0].split()[1]
# # print(type(float(X)))
# print(imagePath)
#
# Im = Image.open(imagePath)
# print(np.array(Im).shape)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])])
     #https://pytorch.org/docs/stable/torchvision/models.html

nEpochs = 10
batchSize = 1
lr = 0.005


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
# transform = transforms.Compose([transform]) transforms.ToTensor() #ToTensor() converts the image from 0~1

train_dataset = Dataset(train_dir, filenames, transform = transform)
# print(type(train_dataset))
# print(train_dataset.__sizeof__())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
# print(len(train_loader))
# images, x, y = next(iter(train_loader))

# model = Resnet_mob(pretrained=True).to(device)
model = ResNet_PC().to(device)
torch.cuda.manual_seed(1)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss().cuda()

# FILEPATH_MODEL_SAVE = ?

FILEPATH_MODEL_LOAD = None

if FILEPATH_MODEL_LOAD is not None:
    train_states = torch.load(FILEPATH_MODEL_LOAD)
    model.load_state_dict(train_states['train_states_latest']['model_state_dict'])
    optimizer.load_state_dict(train_states['train_states_latest']['optimizer_state_dict'])
    train_states_best = train_states['train_states_best']
    loss_valid_min = train_states_best['loss_valid_min']
    model_save_criteria = train_states_best['model_save_criteria']
else:
    train_states = {}
    model_save_criteria = np.inf

loss_train = [] #all epoch
loss_valid = [] #all epoch

for epoch in range(nEpochs):
    running_loss = 0
    epoch_accuracy = 0
    running_time_batch = 0
    time_batch_start = time.time()
    model.train()
    print("training...")
    for tBatchIdx, sample in enumerate(train_loader):
        time_batch_load = time.time() - time_batch_start
        Image = sample[0].float().to(device)
        # print(np.array(Image).shape)   # 4, 3, 326, 490
        # print('Data: ', torch.min(Image), torch.max(Image)) #0~1
        X = sample[1].float().to(device)
        Y = sample[2].float().to(device)
        output = torch.sigmoid(model(Image))
        print(output, X, Y)
        optimizer.zero_grad()
        lossX = criterion(X, output[:, 0])
        print("lossX", lossX)
        lossY = criterion(Y, output[:, 1])
        print("lossY", lossY)
        loss = lossX + lossY
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        mean_loss = running_loss / (tBatchIdx + 1)
        # print time stats
        time_compute = time.time() - time_batch_start
        time_batch = time_batch_load + time_compute
        running_time_batch += time_batch
        time_batch_avg = running_time_batch / (tBatchIdx + 1)

        print(
            'epoch: {}/{}, batch: {}/{}, loss-train: {:.4f}, batch time taken: {:.2f}s, eta_epoch: {:.2f} hours'.format(
                epoch + 1,
                nEpochs,
                tBatchIdx + 1,
                len(train_loader),
                mean_loss,
                time_batch,
                time_batch_avg * (len(train_loader) - (tBatchIdx + 1)) / 3600,
            )
        )
        time_batch_start=time.time()
    loss_train.append(mean_loss)

    ## Validation
    # for i, sample in enumerate(valid_loader):
    #     running_loss = 0
    #     model11.eval()
    #     with torch.no_grad():
    #         mr = sample[0].float().to(device)
    #         abs = sample[1].float().to(device)
    #         output = model11(mr)
    #         loss = criterion(abs, output)
    #         running_loss += loss.item()
    #         mean_loss = running_loss / (i + 1)
    #         print(
    #             'epoch: {}/{}, batch: {}/{}, loss-valid: {:.4f}'.format(
    #                 epoch + 1,
    #                 max_epochs,
    #                 i + 1,
    #                 len(valid_loader),
    #                 mean_loss,
    #             )
    #         )
    # loss_epoch_valid.append(mean_loss)

    ## Save model if loss decreases
    chosen_criteria = mean_loss
    print('criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))

    if chosen_criteria < model_save_criteria:  # save model if true
        print('criteria decreased from {:.4f} to {:.4f}, saving model...'.format(model_save_criteria,
                                                                                 chosen_criteria))
        train_states_best = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_save_criteria': chosen_criteria,
        }
        train_states = {
            'train_states_best': train_states_best,
        }
        # torch.save(train_states, FILEPATH_MODEL_SAVE)

        model_save_criteria = chosen_criteria

    log = {
        'loss_train': loss_train,
        'loss_valid': loss_valid,
    }
    with open(FILEPATH_LOG, 'wb') as pfile:
        pickle.dump(log, pfile)

    ## also save the latest model after each epoch as you may want to resume training at a later time
    train_states_latest = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_save_criteria': chosen_criteria,
    }
    train_states['train_states_latest'] = train_states_latest
    # torch.save(train_states, FILEPATH_MODEL_SAVE)

# ## Test
# for i, sample in enumerate(test_loader):
#     running_loss = 0
#     model11.eval()  # sets the model in evaluation mode
#     with torch.no_grad():
#         mr = sample[0].float().to(device)
#         abs = sample[1].float().to(device)
#         output = model11(mr)
#         loss = criterion(abs, output)
#         running_loss += loss.item()
#         mean_loss = running_loss / (i + 1)
# print('test_loss {:.4f}'.format(mean_loss))
#         # break





# for i, sample in enumerate(train_loader):
#
#     # break
#     predict = model(Image)
#     optimizer.zero_grad()
#     lossX = criterion(X, predict[:, 0])
#     total_loss = lossX + lossY
#     total_loss.backward()
#     optimizer.step()
# Image = np.array(Image)
# print(np.max(Image))

# output = model(Image)

# print(output)

