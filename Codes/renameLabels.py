import os
import glob

dir_label = r'C:\CS8395_DLMIC\data\assignment1_data\labels'

filePath = glob.glob(os.path.join(dir_label, '*ls.txt'))
# print(filePath)
filePath_names = glob.glob(os.path.join(dir_label, '*mes.txt'))
filePath_coos = glob.glob(os.path.join(dir_label, '*tes.txt'))
filePath_x = glob.glob(os.path.join(dir_label, 'Xcoor.txt'))
filePath_y = glob.glob(os.path.join(dir_label, 'Ycoor.txt'))


with open(os.path.join(filePath_coos[0]),'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]

# print(filenames)

flnames = []

# f = open(filePath_coos[0], 'r')
for x in filenames:
#     print(x)
#     # print(x.split()[1]+x.split()[2])
#     flnames.append(x.split()[0])
    g = open(filePath_y[0], 'a')
    g.write('{}\n'.format(x.split()[1]))
# #     g.write(x.split()[1])
# #     g.write(' ')
# #     g.write(x.split()[2])
# #     g.write('\n')
    g.close()
# print(flnames)

# f = open(filePath_names[0],'w')
# f.write(flnames)
# f.close()

# with open(filePath_x, 'w') as file:
#     for fl in flnames:
#         print(x)
#         file.write('{}\n'.format(fl))