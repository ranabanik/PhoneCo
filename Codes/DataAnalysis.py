import numpy as np
import os
import glob
import pickle
from PIL import Image
import matplotlib.pyplot as plt

TIME_STAMP = '2020-01-26-02-40-13'
data_dir = r'C:\CS8395_DLMIC\data\assignment1_data'
train_dir = os.path.join(data_dir, 'train')
ImageFiles = glob.glob(os.path.join(train_dir, '*.jpg'))
dir_log = r'C:\Users\ranab\OneDrive\PycharmProjects\PhoneCo\Project\log'
project_dir_log = os.path.join(dir_log, TIME_STAMP)

if __name__ != '__main__':
    print(ImageFiles)
    MaxMin = []
    for i in ImageFiles:
        Im = Image.open(i)
        Im = np.array(Im)
        MaxMin.append([os.path.basename(i), Im.max(), Im.min()])
        print(os.path.basename(i))

        EDAfile = os.path.join(train_dir, 'EDA.txt')

    with open(EDAfile, 'w') as file:
        for fl in MaxMin:
            file.write('{}\n'.format(fl))

if __name__ == '__main__':
    file = os.path.join(project_dir_log, '{}.bin'.format(TIME_STAMP))
    with open(file, 'rb') as pfile:
        h = pickle.load(pfile)
    print(h.keys())

    plt.plot(h['loss_train'], 'r', linewidth=1.5, label='Training loss')
    plt.plot(h['loss_valid'], 'b', linewidth=1.5, label='Validation loss')
    # print(h['loss_train'], np.min(h['loss_train']))
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss(a.u.)', fontsize=15)
    plt.legend()
    plt.show()


# model = ResNet3D().to(device)
# FILEPATH_MODEL_LOAD = os.path.join(dir_model,'{}.pt'.format(model_time_stamp))