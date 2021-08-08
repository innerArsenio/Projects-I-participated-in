import torch
import torch.autograd.variable as Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import mods as models




PATH_TO_FOLDS = "train_val_txt_files_per_fold"
PATH_TO_IMAGE_FOLDERS = "aligned"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
check = torch.load("checkpoint.pth", map_location=device)

model = models.vgg16(num_classes=8)
model.load_weights(check["state_dict"])



class AdienceDataset(Dataset) :
    def __init__(self, txt_files, root_dir, transform) :
        self.txt_files = txt_files
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.read_from_txt_files()

    def __len__(self) :
        return len(self.data)

    def read_from_txt_file(self,txt_file) :
        data = []
        f = open(txt_file)
        for line in f.readlines() :
            image, label = line.split()
            label = int(label)
            data.append((image, label))
        return data

    def read_from_txt_files(self):
        for txt_file in self.txt_files:
            self.data.extend(self.read_from_txt_file(txt_file))


    def __getitem__(self, idx) :
        img, label = self.data[idx]
        image = Image.open(os.path.join(self.root_dir, img))

        if self.transform :
            image = self.transform(image)

        return {
            'image' : image,
            'label' : label
        }






def get_dataloader( type, category, fold_numbers, transform_index, minibatch_size, num_workers, Gray=False) :
    applied_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomCrop(227),
        transforms.Grayscale(),
        transforms.GaussianBlur(kernel_size=(65, 65)),
        transforms.RandomRotation(20)
    ]

    dictionary_of_transformations = {
        'train' : {
            0 : list(applied_transforms[i] for i in [0, 1, 3]),  # no transformation
            1 : list(applied_transforms[i] for i in [0, 1, 2, 3]),  # random horizontal flip
            2 : list(applied_transforms[i] for i in [0, 4, 2, 3]),  # random crop and random horizontal flip
            3 : list(applied_transforms[i] for i in [0, 1, 6, 3]),
            4 : list(applied_transforms[i] for i in [0, 1, 7, 3]),
            5: list(applied_transforms[i] for i in [0, 1, 2, 5, 3])
        },
        'val' : {
            0 : list(applied_transforms[i] for i in [0, 1, 3]),
        },
        'test' : {
            0 : list(applied_transforms[i] for i in [0, 1, 3]),
        }
    }
    if Gray:
        transform_index = 5
    #s test, val, train
    #c age, gender
    #fold 0-4
    #transform index corresponding to train,test,val
    txt_files = []
    for fold_number in fold_numbers:
        test_fold = 'test_fold_is_' + str(fold_number) + '/' +category + '_' +type+'.txt'
        txt_file = os.path.join(PATH_TO_FOLDS,test_fold)
        txt_files.append(txt_file)
    root_dir = PATH_TO_IMAGE_FOLDERS

    transformed_dataset = AdienceDataset(txt_files, root_dir,
                                         transforms.Compose(dictionary_of_transformations[type][transform_index]))
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
    return transformed_dataset, dataloader





def main():
    _, dataloader = get_dataloader('test', 'age', [1], 0, 64, False)
    norm_conf=confusion_matrix(dataloader)

    # one-off accuracy
    ncls = len(norm_conf)
    tot_acc = 0

    for row in norm_conf:
        for el in row:
            print("%.2f"%(el*100), end="\t")
        print("")

    for i in range(ncls):
        
        acc = norm_conf[i][i]

        # add left
        if i > 0:
            acc += norm_conf[i][i-1]
        
        if i < ncls - 1:
            acc += norm_conf[i][i+1]
        
        tot_acc += acc

    tot_acc = tot_acc / ncls
    print("%.2f" % (tot_acc * 100))





def confusion_matrix(valid_loader):

    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

    cuda_exists = True if torch.cuda.is_available() else False
    # confusion matrix of ncls * ncls
    ncls = len(age_list)
    conf_matrix = torch.zeros(ncls, ncls)

    # switch to evaluate mode
    model.eval()

    
    with torch.no_grad():
        for i, sample in enumerate(valid_loader):
    

            img = sample.get('image')
            target = sample.get('label')

            # compute output
            if cuda_exists:
                img = img.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            res = model(img)
            _, preds = torch.max(res, 1)

            for t, p in zip(target.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1


    # horiz normalization to get percentage
    norm_conf = []
    for row in conf_matrix:
        factor = float(row.sum())
        normed = [float(i) / factor for i in row]
        norm_conf.append(normed)

    return norm_conf




# Start!
main()