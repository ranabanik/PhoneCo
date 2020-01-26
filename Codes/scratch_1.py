import numpy as np
import os
import glob
import time
import copy
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models, transforms
from DataLoader import Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dir = r'C:\CS8395_DLMIC\data\assignment1_data\train'
label_dir = r'C:\CS8395_DLMIC\data\assignment1_data\labels'
label_txt = glob.glob(os.path.join(label_dir, '*els.txt'))

with open(os.path.join(label_txt[0]), 'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# print(num_ftrs)
print(model_ft)
model_ft.fc = nn.Linear(num_ftrs,2)
# print(model_ft)
model_ft = model_ft.to(device)

if __name__ != '__main__()':
    nEpochs = 10
    batchSize = 4
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    train_dataset = Dataset(train_dir, filenames, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    # dataloaders = {x: torch.utils.data.DataLoader(train_dataset[x], batch_size=4,
    #                                              shuffle=True, num_workers=4)
    #               for x in ['train', 'val']}

    # for inputs, labels in enumerate(dataloaders):
    #     print(inputs.shape())

    # def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    #     since = time.time()
    #
    #     best_model_wts = copy.deepcopy(model.state_dict())
    #     best_acc = 0.0
    #
    #     for epoch in range(num_epochs):
    #         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #         print('-' * 10)
    #
    #         # Each epoch has a training and validation phase
    #         for phase in ['train', 'val']:
    #             if phase == 'train':
    #                 model.train()  # Set model to training mode
    #             else:
    #                 model.eval()   # Set model to evaluate mode
    #
    #             running_loss = 0.0
    #             running_corrects = 0
    #
    #             # Iterate over data.
    #             for inputs, labels in dataloaders[phase]:
    #                 inputs = inputs.to(device)
    #                 labels = labels.to(device)
    #
    #                 # zero the parameter gradients
    #                 optimizer.zero_grad()
    #
    #                 # forward
    #                 # track history if only in train
    #                 with torch.set_grad_enabled(phase == 'train'):
    #                     outputs = model(inputs)
    #                     # _, preds = torch.max(outputs, 1)
    #                     loss = criterion(outputs, labels)
    #
    #                     # backward + optimize only if in training phase
    #                     if phase == 'train':
    #                         loss.backward()
    #                         optimizer.step()
    #
    #                 # statistics
    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects += torch.sum(preds == labels.data)
    #             if phase == 'train':
    #                 scheduler.step()
    #
    #             epoch_loss = running_loss / dataset_sizes[phase]
    #             epoch_acc = running_corrects.double() / dataset_sizes[phase]
    #
    #             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
    #                 phase, epoch_loss, epoch_acc))
    #
    #             # deep copy the model
    #             if phase == 'val' and epoch_acc > best_acc:
    #                 best_acc = epoch_acc
    #                 best_model_wts = copy.deepcopy(model.state_dict())
    #
    #         print()
    #
    #     time_elapsed = time.time() - since
    #     print('Training complete in {:.0f}m {:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60))
    #     print('Best val Acc: {:4f}'.format(best_acc))
    #
    #     # load best model weights
    #     model.load_state_dict(best_model_wts)
    #     return model
    loss_train = []

    for epoch in range(nEpochs):
        running_loss = 0
        epoch_accuracy = 0
        running_time_batch = 0
        time_batch_start = time.time()
        model_ft.train()
        print("training...")
        for tBatchIdx, sample in enumerate(train_loader):
            time_batch_load = time.time() - time_batch_start
            Image = sample[0].float().to(device)
            # print(np.array(Image).shape)   # 4, 3, 326, 490
            # print('Data: ', torch.min(Image), torch.max(Image)) #0~1
            X = sample[1].float().to(device)
            Y = sample[2].float().to(device)
            optimizer_ft.zero_grad()
            output = torch.sigmoid(model_ft(Image))
            print(output)
            lossX = criterion(X, output[:, 0])
            # print("lossX", lossX)
            lossY = criterion(Y, output[:, 1])
            # print("lossY", lossY)
            loss = lossX + lossY
            loss.backward()
            optimizer_ft.step()
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