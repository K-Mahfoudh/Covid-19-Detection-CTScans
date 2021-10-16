from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch


class Data:
    """
    Class for managing and preparing data for training, validation and testing phases.
    """

    def __init__(self, train_path, test_path, batch_size, valid_size, shuffle):
        """
        Data class constructor.

        :param train_path: string path to train set
        :param test_path: string path to test set
        :param batch_size: integer representing batch size
        :param valid_size: float representing percentage of data used in validation
        :param shuffle: boolean value (True to shuffle data, False otherwise)
        """
        self._train_path = train_path
        self._test_path = test_path
        self._batch_size = batch_size
        self._valid_size = valid_size
        self._shuffle = shuffle
        self._train_transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @property
    def train_path(self):
        return self._train_path

    @property
    def test_path(self):
        return self._test_path

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def valid_size(self):
        return self._valid_size

    @train_path.setter
    def train_path(self, train_path):
        self._train_path = train_path

    @test_path.setter
    def test_path(self, test_path):
        self._test_path = test_path

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @valid_size.setter
    def valid_size(self, valid_size):
        self._valid_size = valid_size

    def get_trainset(self):
        """
        spliting training data into train and validation sets based on a given validation size (as class attribute).

        :return: dataloader tuple of training and validation sets
        """
        # Loading the data from folder
        trainset = datasets.ImageFolder(self._train_path, transform=self._train_transform)

        # Getting Subset samplers for splitting train and validation data
        train_sampler, valid_sampler = self.train_valid_sampler(trainset)

        # Creating dataLoaders
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self._batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(trainset, batch_size=self._batch_size, sampler=valid_sampler)

        return train_loader, valid_loader

    def get_testset(self):
        """
        loading test set on dataloader object.

        :return: dataloader object containing test set
        """
        # Loading data from folder
        testset = datasets.ImageFolder(self._test_path, transform=self._test_transform)

        # Creating dataLoader
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self._batch_size, shuffle=self._shuffle)

        return test_loader

    def train_valid_sampler(self, data):
        """
        Method used to create samplers for training and validation sets, these samplers will be used as arguments in
        Dataloader constructor.

        :param data: ImageFolder object that will be used for training and validation
        :return: tuple of SubsetRandomSampler instances
        """
        # Getting size of dataset
        data_size = len(data)

        # Creating index list
        index_list = list(range(data_size))

        # Shuffling index_list
        np.random.shuffle(index_list)

        # Creating a splitter
        splitter = int(np.floor(self._valid_size * data_size))

        # Creating separated train and validation index lists
        train_index_list, valid_index_list = index_list[splitter:], index_list[:splitter]

        # Creating Random samplers
        train_sampler, valid_sampler = SubsetRandomSampler(train_index_list), SubsetRandomSampler(valid_index_list)

        return train_sampler, valid_sampler
