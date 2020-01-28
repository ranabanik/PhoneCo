import numpy as np
import os
import glob
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# /home/banikr/PycharmProjects/Project/PhoneCo/log/2020-01-26-22-17-46
TIME_STAMP = '2020-01-27-21-41-55'#'2020-01-27-21-16-33'#'2020-01-27-21-10-12'#'2020-01-27-15-39-28'#'2020-01-27-15-34-02' #'2020-01-27-03-07-53'#'2020-01-26-22-17-46'
data_dir = r'/media/banikr/DATA/CS8395assignment1_data'  #todo: change it
train_dir = os.path.join(data_dir, 'train')
ImageFiles = glob.glob(os.path.join(train_dir, '*.jpg'))
dir_log = r'/home/banikr/PycharmProjects/Project/PhoneCo/log'
# project_dir_log = os.path.join(dir_log, TIME_STAMP)

if __name__ != '__main__':
    print(ImageFiles)
    MaxMin = []
    for i in ImageFiles:
        Im = Image.open(i)
        Im = np.array(Im)
        MaxMin.append([os.path.basename(i), Im.shape])
        print(os.path.basename(i))

        EDAfile = os.path.join(train_dir, 'EDA2.txt')

    with open(EDAfile, 'w') as file:
        for fl in MaxMin:
            file.write('{}\n'.format(fl))

if __name__ == '__main__':
    file = os.path.join(dir_log, TIME_STAMP, '{}.bin'.format(TIME_STAMP))
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