from __future__ import division
import sys
import json
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import namedtuple
from os.path import join
import numpy as np

RNNParams = namedtuple('RNNParams', 'sequence_length, input_size, hidden_size, num_layers, num_classes, batch_size, num_epochs, learning_rate, dropout')

use_cuda = torch.cuda.is_available()
gpu_id = 0

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def load_state_dict(self, state_dict):
        super(RNN, self).load_state_dict(state_dict)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        if use_cuda:
            h0 = h0.cuda(gpu_id)
            c0 = c0.cuda(gpu_id)
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

    rnn = RNN(rnn_params.input_size, rnn_params.hidden_size, rnn_params.num_layers, rnn_params.num_classes, rnn_params.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=rnn_params.learning_rate)

    if use_cuda:
        rnn = rnn.cuda(gpu_id)
        criterion = criterion.cuda(gpu_id)

    epoch_times = []
    for epoch in range(rnn_params.num_epochs):
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, rnn_params.sequence_length, rnn_params.input_size))
            if use_cuda:
                images = images.cuda(gpu_id)
            optimizer.zero_grad()
            outputs = rnn(images)
            labels = Variable(labels)
            if use_cuda:
                outputs = outputs.cuda(gpu_id)
                labels = labels.cuda(gpu_id)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                avg_time = 0 if len(epoch_times) == 0 else np.mean(epoch_times)
                print 'Epoch [%d/%d], Step [%d/%d], Avg. Time: %.3f [s] , Loss: %.8f' % (
                epoch + 1, rnn_params.num_epochs, i + 1, len(train_dataset) / rnn_params.batch_size,  avg_time,loss.data[0])
        epoch_times.append(time.time()-start_time)

        if epoch % 1 == 0 and save_path:
            model_name = 'rnn_%04d.pkl'%(epoch)
            torch.save(rnn.state_dict(), join(save_path, model_name))

    return rnn

def eval(rnn, test_dataset, rnn_params):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=rnn_params.batch_size,
                                              shuffle=False)
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, rnn_params.sequence_length, rnn_params.input_size))
        if use_cuda:
            images = images.cuda(gpu_id)
        outputs = rnn(images)

        if use_cuda:
            labels = labels.cuda(gpu_id)
            outputs = outputs.cuda(gpu_id)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted.view(-1) == labels.view(-1)).sum()
    print 'Test Accuracy of the model on the %d test images: %d%%' % (len(test_dataset),100 * correct / total)

def test(state_dict_path, test_dataset, rnn_params):
    rnn = RNN(rnn_params.input_size, rnn_params.hidden_size, rnn_params.num_layers, rnn_params.num_classes)
    rnn.load_state_dict(torch.load(state_dict_path))
    eval(rnn, test_dataset, rnn_params)

def set_gpu_id(id):
    global gpu_id
    gpu_id = id

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: train_rnn.py <config_path>'
        sys.exit(-1)
    config = json.load(open(sys.argv[1]))

    if use_cuda and 'gpu' in config:
        set_gpu_id(config['gpu'])
        print 'Using GPU: {}'.format(gpu_id)

    params = RNNParams(sequence_length = config['rnn']['seq_len'],
                        input_size = config['rnn']['input'],
                        hidden_size = config['rnn']['hidden'],
                        num_layers = config['rnn']['layers'],
                        num_classes = config['rnn']['num_class'],
                        batch_size = config['rnn']['batch_sz'],
                        num_epochs = config['rnn']['epochs'],
                        learning_rate = config['rnn']['lr'],
                        dropout=config['rnn']['dropout'])

    if config['mode'] == 'train':
        print '>>> Loading the training data <<<'
        train_dataset = load_ucf_dataset(config['train_path'])
        print '>>> Training the model <<<'
        rnn_model = train(train_dataset,  params, config['models_dir'])

        print '>>> Loading the testing data <<<'
        test_dataset = load_ucf_dataset(config['test_path'])
        print '>>> Evaluating the model <<<'
        eval(rnn_model, test_dataset, params)

    elif config['mode'] == 'test':
        use_cuda = False
        test_dataset = load_ucf_dataset(config['test_path'])

        print '>>> Testing the model <<<'
        test(config['model'], test_dataset, params)
