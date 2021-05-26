from torch import nn, optim
from torchvision import models
import torch
import torch.nn.functional as F
import numpy as np
import sys


class Network(nn.Module):
    def __init__(self, model_path,lr):
        super(Network, self).__init__()
        self.model = models.resnext101_32x8d(pretrained=True, progress=True)
        self.criterion = nn.NLLLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_classifier()
        self.set_optimizer('Adam', lr)
        self.model_path = model_path


    def set_classifier(self):
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 2),

            nn.LogSoftmax(dim=1)
        )

    def set_optimizer(self, optimizer, lr):

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.fc.parameters(),lr=lr)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
        else:
            print('Wrong optimizer, please Write \'SGD\' or \'Adam\'')

    def forward(self, data):
        return self.model(data)

    def train_network(self, trainset, validset, epochs):
        # Transfering model to GPU if cuda is available
        self.model.to(self.device)

        # Setting minimum validation loss
        min_valid_loss = np.inf

        # init loss and accuracy list for visualisation
        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []

        for epoch in range(epochs):
            print('Epoch {} Started\r'.format(epoch))
            train_accuracy = 0
            train_loss = 0
            valid_accuracy = 0
            valid_loss = 0

            # Enable training mode
            if not self.model.train():
                self.model.train()

            # training on trainset
            for index, (images, labels) in enumerate(trainset):

                # Transferring images and labels to GPU if available
                images, labels = images.to(self.device), labels.to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.forward(images)

                # Calculating loss
                loss = self.criterion(logits, labels)
                train_loss += loss

                sys.stdout.write('Train Batch {} ==> Loss: {:.3f}\r'.format(index, loss))
                sys.stdout.flush()

                # Backward propagation
                loss.backward()

                # Updating weights
                self.optimizer.step()

                # Getting predictions
                preds = F.softmax(logits, dim=1)

                # getting accuracy
                _, top_classes = preds.topk(1, dim=1)
                compare = top_classes == labels.view(*top_classes.shape)
                train_accuracy += torch.mean(compare.type(torch.FloatTensor))

            # Enable evaluation mode
            if not self.model.eval():
                self.model.eval()

            # Testing on validation set
            with torch.no_grad():
                for index, (images, labels) in enumerate(validset):
                    sys.stdout.write('Validation Batch {}\r'.format(index))
                    sys.stdout.flush()
                    # Transfering data to GPU if available
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    logits = self.forward(images)

                    # Calculating the loss
                    loss = self.criterion(logits, labels)
                    valid_loss += loss

                    # getting predictions
                    preds = F.softmax(logits, dim=1)
                    _, top_classes = preds.topk(1, dim=1)
                    compare = top_classes == labels.view(*top_classes.shape)
                    valid_accuracy += torch.mean(compare.type(torch.FloatTensor))

            # Getting overall accuracy and loss
            train_accuracy = train_accuracy/len(trainset)
            train_loss = train_loss/len(trainset)
            valid_accuracy = valid_accuracy/len(validset)
            valid_loss = valid_loss/len(validset)

            # Appending loss and accuracy to lists for visualisation
            train_accuracy_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            valid_accuracy_list.append(valid_accuracy)
            valid_loss_list.append(valid_loss)

            # Checking for best model
            if valid_loss < min_valid_loss:
                print('Validation loss decreased from {:.3f} ======> {}\r'.format(min_valid_loss, valid_loss))

                min_valid_loss = valid_loss

                # Saving the model
                torch.save(self.model.state_dict(), self.model_path)

            print('Epoch: {} =====>  Train Accuracy: {:.3f} ------ Train Loss: {:.3f} ------ Valid Accuracy: {:.3f} ------ Valid Loss: {:.3f} \r'.format(
                                                                                                epochs,
                                                                                                train_accuracy,
                                                                                                train_loss,
                                                                                                valid_accuracy,
                                                                                                valid_loss))

        return train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list

    def predict(self, data):
        self.load_model()
        if not self.model.eval():
            self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            accuracy = 0
            loss = 0
            for images, labels in iter(data):

                # Transfering data to gpu if available
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                logits = self.forward(images)
                loss += self.criterion(logits, labels)

                # Getting predictions
                preds = F.softmax(logits, dim=1)

                # Calculating accuracy
                top_p, top_class = preds.topk(1, dim=1)
                print(top_class.view(1,-1))
                print(top_p.view(1,-1))

                compare = top_class == labels.view(*top_class.shape)
                print(compare.view(1,-1))
                print(torch.mean(compare.type(torch.FloatTensor)).view(1,-1))
                accuracy += torch.mean(compare.type(torch.FloatTensor))
            accuracy = accuracy/len(data)*100
            loss = loss/len(data)
            print('Accuracy is {:.3f}%'.format(accuracy))
            print('Loss is {:.3f}'.format(loss))
        self.model.train()

    def load_model(self):
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)