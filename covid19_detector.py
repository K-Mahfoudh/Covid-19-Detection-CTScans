from data import Data
from network import Network
import matplotlib.pyplot as plt


def main():

    batch_size = 4
    valid_size = 0.3
    shuffle = True
    model_path = 'models/model5.pth'
    lr = 0.0002
    epochs = 50

    # Creating data Object
    print('loading data')
    data = Data('data/covid_dataset/train', 'data/covid_dataset/test', batch_size, valid_size, shuffle)
    print('data Loaded')

    # Getting train and validation loaders
    #print('Splitting data')
    train_loader, valid_loader = data.get_trainset()

    #test_loader = data.get_testset()
    network = Network(model_path, lr)

    #network.predict(test_loader)
    train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list = network.train_network(train_loader, valid_loader, epochs)

    plt.plot(train_accuracy_list)
    plt.plot(train_loss_list)
    plt.plot(valid_accuracy_list)
    plt.plot(valid_loss_list)
    plt.show()

if __name__ == '__main__':
    main()