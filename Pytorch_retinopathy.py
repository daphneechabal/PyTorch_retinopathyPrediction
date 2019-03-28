#import data
from DiabetesDataset import *
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampler import *
import pickle
from PIL import Image

#split data
def indices_of_extremes(dataset):
    """ Returns all data points that have a severity of 0 or 4.

        This means all images with no diabetic eye disease and
        images with very extreme diabetic eye disease.
    """
    indices = list()

    for idx, data_point in enumerate(dataset):

        if data_point['severity'] in [0, 1, 2, 3, 4]:
            indices.append(idx)

    return indices
vanilla_dataset = DiabetesDataset(csv_file='train.csv',
                                      root_dir='train/',
                                      transform=None)
count_class(vanilla_dataset)

#load data pytorch
vanilla_dataset = DiabetesDataset(csv_file='train.csv',
                                  root_dir='train/',
                                  transform=None)

filtered_dataset_samples = indices_of_extremes(vanilla_dataset)
size_of_dataset = len(filtered_dataset_samples)
validation_size = int(size_of_dataset * 0.2)

validation_samples = np.random.choice(filtered_dataset_samples, size=validation_size, replace=False)
train_samples = list(set(filtered_dataset_samples) - set(validation_samples))

transformed_dataset =  DiabetesDataset(csv_file='train.csv',
                                      root_dir='train/',
                                      transform = transforms.Compose([


                                      transforms.Scale(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))



train_sampler = ImbalancedDatasetSampler(transformed_dataset, train_samples)
print('made train sampler')

validation_sampler = ImbalancedDatasetSampler(transformed_dataset, validation_samples)
print('made validation sampler')

#data loader trainer

trainer_loader = torch.utils.data.DataLoader(transformed_dataset,
                batch_size=60, sampler=train_sampler)


pickle.dump(trainer_loader, open("train_loader_final_60.p", "wb" ))

validation_loader = torch.utils.data.DataLoader(transformed_dataset,
                batch_size=60, sampler=validation_sampler)

pickle.dump(trainer_loader, open("validation_loader_final.p", "wb" ))

#train model

vgg11 = models.resnet18()
vgg11.classifier[-1] = nn.Linear(4096, 5)
vgg11.fc = nn.Linear(512, 5)


def train(model, train_loader, optimizer, criterion):
    """ Trains the network. """

    # Puts the network into training mode.
    model.train()
    model.cuda()

    final_batchidx = 0
    running_loss = 0

    for epoch in range(10):

        print('Current epoch: {}', epoch)
        # For each batch in the training data, ik
        for batch_idx, data in enumerate(train_loader):

            # Zero the gradients, in other words, assume we
            # don't know the next best adjustment to the weights
            optimizer.zero_grad()

            # Use the model for inference on data.
            # In other words, apply the network to an image.
            output = model(Variable(data['image']).cuda())

            # Determine the loss based on the discrepancy
            # between predicted severity class and actual class.
            ground_truth = Variable(data['severity']).cuda()

            loss = criterion(output, ground_truth)

            # Propagate the loss backwards through the network,
            # thereby determining how much each part of the network
            # is "responsible" for the loss. This is of course the
            # gradients of the loss with respect to weights.
            loss.backward()

            # Optimize the network by taking the gradients
            # adjusting the weights accordingly.
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0 and batch_idx != 0:
                print('[%d] loss: %.3f' % (batch_idx, running_loss / 10))

                running_loss = 0.0

        final_batchidx = batch_idx
        print('saving model now')
        torch.save(model.state_dict(), 'model' + str(epoch)+ '.pmod')
        print('model saved!!!')


optimizer = optim.SGD(vgg11.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()
train(vgg11, data_loader, optimizer, criterion)

#Test results

answer_dict = dict()

def inference(model, train_loader, answer_dict):
    """ Trains the network. """

    model.cuda()
    for batch_idx, data in enumerate(train_loader):

        output = model(Variable(data['image']).cuda())
        output = torch.argmax(output, dim=1)

        for idx, name in enumerate(data['name']):
            real_name = name.split('/')[-1]
            real_name = real_name.split('.')[0]
            answer_dict[real_name] = str(int(output[idx]))

    if batch_idx % 10 == 0:
        print (batch_idx)

    return answer_dict


answer_dict = inference(vgg11, data_loader, answer_dict)
print('Finished inferring')
dataframe = pd.DataFrame.from_dict(data=answer_dict, orient='index')
dataframe.to_csv('solution.csv', header=False)
