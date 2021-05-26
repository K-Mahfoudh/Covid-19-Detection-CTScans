from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch


class Data:
    def __init__(self,train_path, test_path, batch_size, valid_size, shuffle):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def set_train_path(self, train_path):
        self.train_path = train_path

    def set_test_path(self, test_path):
        self.test_path = test_path

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_valid_size(self, valid_size):
        self.valid_size = valid_size

    def get_trainset(self):

        # Loading the data from folder
        trainset = datasets.ImageFolder(self.train_path, transform=self.train_transform)

        # Getting Subset samplers for splitting train and validation data
        train_sampler, valid_sampler = self.train_valid_sampler(trainset, self.valid_size)

        # Creating dataLoaders
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=valid_sampler)

        return train_loader, valid_loader

    def get_testset(self):

        # Loading data from folder
        testset = datasets.ImageFolder(self.test_path, transform=self.test_transform)

        # Creating dataLoader
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=self.shuffle)

        return test_loader


    def train_valid_sampler(self, data, valid_size):

        # Getting size of dataset
        data_size = len(data)

        # Creating index list
        index_list = list(range(data_size))

        # Shuffling index_list
        np.random.shuffle(index_list)

        # Creating a splitter
        splitter = int(np.floor(valid_size*data_size))

        # Creating separated train and validation index lists
        train_index_list, valid_index_list = index_list[splitter:], index_list[:splitter]

        # Creating Random samplers
        train_sampler, valid_sampler = SubsetRandomSampler(train_index_list), SubsetRandomSampler(valid_index_list)

        return train_sampler, valid_sampler