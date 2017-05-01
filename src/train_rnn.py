from __future__ import division
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import namedtuple
import numpy as np

RNNParams = namedtuple('RNNParams', 'sequence_length, input_size, hidden_size, num_layers, num_classes, batch_size, num_epochs, learning_rate')

use_cuda = torch.cuda.is_available()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def load_state_dict(self, state_dict):
        super(RNN, self).load_state_dict(state_dict)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def build_mnist_dataset():
    train_dataset = dsets.MNIST(root='../data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='../data/',
                               train=False,
                               transform=transforms.ToTensor())

    return train_dataset, test_dataset


def load_ucf_dataset(data_path):
    data = np.load(data_path)['data']
    labels = np.load(data_path)['labels']

    # Load into Pytorch types
    ucf_samples = torch.from_numpy(data)
    ucf_samples = ucf_samples.float()  # Key-factor to train network
    ucf_labels = torch.from_numpy(labels)

    # build pytorch tensor dataset
    ucf_dataset = torch.utils.data.TensorDataset(ucf_samples, ucf_labels)

    return ucf_dataset

def train(train_dataset, rnn_params, save_path=''):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=rnn_params.batch_size,
                                               shuffle=True)

    rnn = RNN(rnn_params.input_size, rnn_params.hidden_size, rnn_params.num_layers, rnn_params.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=rnn_params.learning_rate)

    if use_cuda:
        rnn.cuda()

    for epoch in range(rnn_params.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, rnn_params.sequence_length, rnn_params.input_size))
            if use_cuda:
                images.cuda()
            optimizer.zero_grad()
            outputs = rnn(images)
            labels = Variable(labels)
            if use_cuda:
                outputs.cuda()
                labels.cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                epoch + 1, rnn_params.num_epochs, i + 1, len(train_dataset) / rnn_params.batch_size, loss.data[0])
                if save_path:
                    torch.save(rnn.state_dict(), save_path)

    return rnn

def eval(rnn, test_dataset, rnn_params):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=rnn_params.batch_size,
                                              shuffle=False)
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, rnn_params.sequence_length, rnn_params.input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print 'Test Accuracy of the model on the %d test images: %d%%' % (len(test_dataset),100 * correct / total)


if __name__ == '__main__':
    params = RNNParams(sequence_length = 180,
                        input_size = 54,
                        hidden_size = 1024,
                        num_layers = 3,
                        num_classes = 12,
                        batch_size = 200,
                        num_epochs = 1000,
                        learning_rate = 0.01)

    print '>>> Loading datasets <<<'
    train_dataset = load_ucf_dataset('/Users/zal/CMU/Spring2017/16824/FinalProject/Data/human_motion.npz')

    print '>>> Training the model <<<'
    rnn_model = train(train_dataset,  params)

    print '>>> Evaluating the model <<<'
    eval(rnn_model, train_dataset, params)