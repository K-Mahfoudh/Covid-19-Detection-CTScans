import sys

from data import Data
from network import Network
import matplotlib.pyplot as plt
import argparse


def main(args):
    # Creating data instance
    data = Data(args.train_path, args.test_path,
                args.batch_size,
                args.valid_size,
                args.shuffle)

    # Creating network instance
    network = Network(args.model_path, args.learning_rate)

    if args.mode in ['Train', 'train']:
        # Getting train and validation loaders
        train_loader, valid_loader = data.get_trainset()
        train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list = network.train_network(train_loader,
                                                                                                           valid_loader,
                                                                                                           args.epochs)
        if args.plot:
            plt.plot(train_accuracy_list)
            plt.plot(train_loss_list)
            plt.plot(valid_accuracy_list)
            plt.plot(valid_loss_list)
            plt.show()

    elif args.mode in ['Test', 'test']:
        test_loader = data.get_testset()
        network.predict(test_loader)


def argument_parser(argv):
    """
    Creating a parser in order to use command line arguments.

    :param argv: arguments to be parsed
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('COVID-19 detection using lungs CT-Scans images classification',
                                     epilog='Note that all arguments are optional, and in case of their absence,'
                                            ' default values will be used.')
    parser.add_argument('-m', '--mode',
                        choices=['Train', 'Test', 'train', 'test'],
                        type=str,
                        default='Train',
                        help='Select Train mode to train a new classifier. Or Test mode to test the model')
    parser.add_argument('-p', '--plot', help='If this argument is passed, the accuracy and loss of both validation '
                                             'and test sets will be plotted')
    parser.add_argument('-tp', '--train_path', type=str, default='data/covid_data/train', help='Path to training '
                                                                                               'dataset')
    parser.add_argument('-ts', '--test_path', type=str, default='data/covid_data/test', help='Path to test set')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size to be used in training')
    parser.add_argument('-vs', '--valid_size',
                        type=float,
                        default=0.3,
                        help='Choosing a float value between 0.0 and 1.0 that will determine the size of '
                             'validation set')
    parser.add_argument('-sh', '--shuffle',
                        type=bool,
                        default=True,
                        help='Used as argument to DataLoader constructor that determines whether we want to shuffle '
                             'data or not.')
    parser.add_argument('--model_path',
                        type=str,
                        default='models/model.pth',
                        help='Path used to save the model while training, or to load an existent model for testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate to be used in training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs used to train the model')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(argument_parser(sys.argv[1:]))
