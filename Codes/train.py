import numpy as np
import os
import glob
import time
import pickle
import copy
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import torch.nn as nn
from torchvision import models, transforms
from DataLoader import Dataset
from model import Resnet_mob

class Dataset(data.Dataset):
    def __init__(self, imageDir, imageList, transform = None): #imageDir can be either train, test or validation"""
        self.imageDir = imageDir
        self.imageList = imageList #with or without label?
        self.transform = transform
        # self.imagePaths = glob.glob(os.path.join(self.imageList))

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, item):
        image_path = os.path.join(self.imageDir, self.imageList[item].split()[0])
        Im = Image.open(image_path).convert("RGB")

        X = self.imageList[item].split()[1]
        X = float(X)
        # print("X: ", X, type(X))
        X = torch.tensor(X)
        # X = ToTensor()(np.array(X))
        Y = self.imageList[item].split()[2]
        Y = float(Y)
        Y = torch.tensor(Y)

        if self.transform is not None:
            Im = self.transform(Im)
        # print("Min: ", torch.min(Im))
        return Im, X, Y

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
project_dir = r'/home/banikr/PycharmProjects/Project/PhoneCo'

if __name__ == '__main__':
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    dir_log = os.path.join(project_dir, 'log')
    train_dir = r'/media/banikr/DATA/CS8395assignment1_data/train'
    valid_dir = r'/media/banikr/DATA/CS8395assignment1_data/validation'
    label_dir = r'/media/banikr/DATA/CS8395assignment1_data/labels'
    train_labels = glob.glob(os.path.join(label_dir, 'trainlabels.txt'))
    valid_labels = glob.glob(os.path.join(label_dir, 'validlabels.txt'))
    FILEPATH_MODEL_SAVE = os.path.join(project_dir, '{}.pt'.format(TIME_STAMP))
    this_Project_log = os.path.join(dir_log, TIME_STAMP)
    os.mkdir(this_Project_log)
    FILEPATH_LOG = os.path.join(this_Project_log, '{}.bin'.format(TIME_STAMP))

    with open(os.path.join(train_labels[0]), 'r') as f:
        trainImages = f.readlines()
    trainImages = [item.strip() for item in trainImages]
    print("Number of train images:", len(trainImages))

    with open(os.path.join(valid_labels[0]), 'r') as f:
        validImages = f.readlines()
    validImages = [item.strip() for item in validImages]

    model_ft = models.resnet18(pretrained=True).to(device)
    # model = Resnet_mob(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # print(num_ftrs)
    # print(model_ft,model)
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = nn.Sequential(model_ft, nn.Linear(2,2))
    print(model_ft)
    lr = 0.001
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # print(model_ft)
    # print(model)
    model_ft = model_ft.to(device)
    # best_model_weights = copy.deepcopy(model_ft.state_dict())
    # model_save_criteria = np.inf

    FILEPATH_MODEL_LOAD = None
    if FILEPATH_MODEL_LOAD is not None:
        train_states = torch.load(FILEPATH_MODEL_LOAD)
        model_ft.load_state_dict(train_states['train_states_latest']['model_state_dict'])
        optimizer_ft.load_state_dict(train_states['train_states_latest']['optimizer_state_dict'])
        train_states_best = train_states['train_states_best']
        loss_valid_min = train_states_best['loss_valid_min']
        model_save_criteria = train_states_best['model_save_criteria']
    else:
        train_states = {}
        model_save_criteria = np.inf

    nEpochs = 100
    batchSize = 10

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    train_dataset = Dataset(train_dir, trainImages, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    valid_dataset = Dataset(valid_dir, validImages, transform=transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchSize, shuffle=False)

    loss_train = []
    loss_valid = []

    for epoch in range(nEpochs):
        running_loss = 0
        # epoch_accuracy = 0
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
            # output = model_ft(Image)  #Sigmoid in sequential
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

            print('epoch: {}/{}, batch: {}/{}, loss-train: {:.4f}, batch time taken: {:.2f}s, eta_epoch: {:.2f} hours'.format(
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

        # exp_lr_scheduler.step() #todo: where it should be?

        running_loss = 0
        for vBatchidx, sample in enumerate(valid_loader):
            model_ft.eval()
            print("Validation...")
            with torch.no_grad():
                Image = sample[0].float().to(device)
                X = sample[1].float().to(device)
                Y = sample[2].float().to(device)
                output = torch.sigmoid(model_ft(Image))
                lossX = criterion(X, output[:, 0])
                lossY = criterion(Y, output[:, 1])
                loss = lossX + lossY
                running_loss += loss.item()
                mean_loss = running_loss / (vBatchidx + 1)
                print(
                    'epoch: {}/{}, batch: {}/{}, loss-valid: {:.4f}'.format(
                        epoch + 1,
                        nEpochs,
                        vBatchidx + 1,
                        len(valid_loader),
                        mean_loss,
                    )
                )
        loss_valid.append(mean_loss)
        chosen_criteria = mean_loss
        print('criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))

        if chosen_criteria < model_save_criteria:  # save model if true
            print('criteria decreased from {:.4f} to {:.4f}, saving model...'
                  .format(model_save_criteria, chosen_criteria))

            # best_model_wts = copy.deepcopy(model_.state_dict())
            train_states_best = {
                'epoch': epoch + 1,
                'model_state_dict': model_ft.state_dict(),
                'optimizer_state_dict': optimizer_ft.state_dict(),
                'model_save_criteria': chosen_criteria,
            }

            train_states['train_states_best'] = train_states_best
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
            'model_state_dict': model_ft.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'model_save_criteria': chosen_criteria,
        }
        train_states['train_states_latest'] = train_states_latest
        torch.save(train_states, FILEPATH_MODEL_SAVE)

    FILEPATH_config = os.path.join(this_Project_log, 'config.txt')
    with open(FILEPATH_config, 'w') as file:
        file.write('Batch size:{}\n'
                   'Epochs:{}\n'
                   'Learning rate:{}\n'
                   'Cross validation #folds:{}\n'
                   'Criterion:{}\n'
                   'Optimizer:{}\n'
                   'Network architecture(layers used):\n{}'.format(batchSize, nEpochs, lr, 0,
                                                                   criterion, optimizer_ft, model_ft))

print(TIME_STAMP)
