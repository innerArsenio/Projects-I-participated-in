import torch
import mods
import helper
import os
import numpy

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision import transforms

trainTransform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.437, 0.340, 0.304], std=[0.286, 0.252, 0.236]),
    ]
)

validTransform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.437, 0.340, 0.304], std=[0.286, 0.252, 0.236]),
    ]
)

usingCuda = True if torch.cuda.is_available() else False

trainDataset = datasets.ImageFolder("faces", trainTransform)
validDataset = datasets.ImageFolder("faces", validTransform)


indices = list(range(len(trainDataset)))
split = int(0.2 * len(trainDataset))

numpy.random.shuffle(indices)


tIdx, vIdx = indices[split:], indices[:split]
trainSampler = SubsetRandomSampler(tIdx)
validSampler = SubsetRandomSampler(vIdx)


train_loader = DataLoader(
    trainDataset,
    batch_size=32,
    sampler=trainSampler,
    num_workers=0,
    pin_memory=usingCuda,
)
valid_loader = DataLoader(
    validDataset,
    batch_size=32,
    sampler=validSampler,
    num_workers=0,
    pin_memory=usingCuda,
)



device = "cuda:0" if usingCuda else "cpu"
weightsVGG=torch.load("vgg_face_dag.pth", map_location=device)
 
#VGG 16 weights
model = mods.vgg16(num_classes=8)
model.load_weights(weightsVGG)

#training and saving weights as checkpoint.pth
helper.train(model, (train_loader, valid_loader))

#loading weights after training
#check = torch.load("checkpoint.pth", map_location=device)
#model.load_weights(check["state_dict"])

