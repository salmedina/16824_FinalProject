from __future__ import division
import sys
import json
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import namedtuple
from os.path import join
import numpy as np

# GLOBALS
RNNParams = namedtuple('RNNParams', 'name, sequence_length, input_size, hidden_size, num_layers, num_classes, batch_size, num_epochs, learning_rate, dropout')
use_cuda = torch.cuda.is_available()
gpu_id = 0

def set_gpu_id(id):
    global gpu_id
    gpu_id = id

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

def load_model(state_dict_path, rnn_params):
    rnn = RNN(rnn_params.input_size, rnn_params.hidden_size, rnn_params.num_layers, rnn_params.num_classes)
    rnn.load_state_dict(torch.load(state_dict_path))
    return rnn

def build_mnist_dataset():
    train_dataset = dsets.MNIST(root='../data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='../data/',
                               train=False,
                               transform=transforms.ToTensor())

    return train_dataset, test_dataset

def calc_label_weights(labels):
    # Easier to use count method
    labels_list = list(labels)
    counts = [labels_list.count(x) for x in range(max(labels_list)+1)]
    # Calculate the ratio wrt to largest value
    max_count = max(counts)
    ratio = [max_count/c for c in counts]
    # Calculate the weights by normalizing the ratios
    sum_ratio = sum(ratio)
    label_weights = [r/sum_ratio for r in ratio]
    weights = [label_weights[l] for l in labels_list]
    return weights

def build_weighted_sampler(data_path):
    labels = np.load(data_path)['labels']
    weights = calc_label_weights(labels)
    assert len(weights) == len(labels)

    return sampler.WeightedRandomSampler(weights, len(labels), replacement=True)

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

def train(data_path, rnn_params, save_path=''):
    print '>>> Loading the training data <<<'
    train_dataset = load_ucf_dataset(data_path)
    weighted_sampler = build_weighted_sampler(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=rnn_params.batch_size,
                                               shuffle=True,
                                               sampler=weighted_sampler)

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
                epoch+1, rnn_params.num_epochs, i+1, len(train_dataset) / rnn_params.batch_size,  avg_time,loss.data[0])
        epoch_times.append(time.time()-start_time)

        if epoch % 1 == 0 and save_path:
            model_name = '%s_%04d.pkl'%(rnn_params.name, epoch+1)
            torch.save(rnn.state_dict(), join(save_path, model_name))

    return rnn

def test(rnn, test_dataset, rnn_params):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=rnn_params.batch_size,
                                              shuffle=False)
    correct = 0
    total = 0
    if use_cuda:
        rnn = rnn.cuda(gpu_id)

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

    acc = correct / total
    return acc

def eval(rnn, data_path, rnn_params):
    print '>>> Loading the testing data <<<'
    test_dataset = load_ucf_dataset(data_path)
    acc = test(rnn, test_dataset, rnn_params)
    print 'Test Accuracy of the model on the %d test clips: %0.4f' % (len(test_dataset), acc)
    return acc

def batch_eval(model_path_tpl, min_epoch, max_epoch, data_path, rnn_params, eval_step=1):
    print '>>> Loading the testing data <<<'
    test_dataset = load_ucf_dataset(data_path)
    for epoch in range(min_epoch, max_epoch+1, eval_step):
        model_path = model_path_tpl % (epoch)
        rnn_model = load_model(model_path, rnn_params)
        print 'Testing model {}'.format(model_path)
        acc = test(rnn_model, test_dataset, rnn_params)
        print 'Epoch: %d, Acc: %0.4f' % (epoch, acc)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: train_rnn.py <config_path>'
        sys.exit(-1)
    config = json.load(open(sys.argv[1]))

    # Configure according to GPU settings
    # if gpu is not mentioned in the config file, fall back to CPU
    if 'gpu' not in config:
        use_cuda = False
    # Otherwise, if there is hardware available, use the one specified by the user
    if use_cuda:
        set_gpu_id(config['gpu'])
        print 'Using GPU: {}'.format(gpu_id)

    # Contain the RNN params in one structure
    params = RNNParams(sequence_length = config['rnn']['seq_len'],
                        input_size = config['rnn']['input'],
                        hidden_size = config['rnn']['hidden'],
                        num_layers = config['rnn']['layers'],
                        num_classes = config['rnn']['num_class'],
                        batch_size = config['rnn']['batch_sz'],
                        num_epochs = config['rnn']['epochs'],
                        learning_rate = config['rnn']['lr'],
                        dropout=config['rnn']['dropout'],
                        name=config['rnn']['name'])

    # TRAINING MODE
    if config['mode'] == 'train':
        print '>>> Training the model <<<'
        rnn_model = train(config['train_path'],  params, config['models_dir'])
        print '>>> Evaluating the model <<<'
        eval(rnn_model, config['test_path'], params)

    # TESTING MODE
    elif config['mode'] == 'test':
        print '>>> Loading the model <<<'
        rnn_model = load_model(config['model'], params)
        print '>>> Testing the model <<<'
        eval(rnn_model, config['test_path'], params)
    # BATCH TEST
    elif config['mode'] == 'batch_test':
        min_epoch = 1
        if 'min_epoch' in config:
            min_epoch = int(config['min_epoch'])
        max_epoch = int(config['max_epoch'])
        batch_step = int(config['batch_step'])
        batch_eval(config['model'], min_epoch, max_epoch, config['test_path'], params, batch_step)
