import unidecode
import string
import random
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time,math

# Helpers for creating training data
def random_chunk(file, chunk_len):
    start_index = random.randint(0, len(file) - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def char_tensor(string, all_characters):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_training_set(file, chunk_len, all_characters):    
    chunk = random_chunk(file, chunk_len)
    inp = char_tensor(chunk[:-1], all_characters)
    target = char_tensor(chunk[1:], all_characters)
    return inp, target

# The neural network model 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

def evaluate(decoder, all_characters, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str, all_characters)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char, all_characters)

    return predicted

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(decoder, criterion, decoder_optimizer, chunk_len, inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / chunk_len

def train_net(args):
    all_characters = string.printable
    n_characters = len(all_characters)

    file = unidecode.unidecode(open(args.filename).read())
    file_len = len(file)
    print('file length = %d' %file_len)

    n_epochs = args.n_epochs
    print_every = args.print_every
    plot_every = 10
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    lr = args.learning_rate
    chunk_len = args.chunk_len

    decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    all_losses = []
    loss_avg = 0

    for epoch in range(1, n_epochs + 1):
        loss = train(decoder, criterion, decoder_optimizer, chunk_len, *random_training_set(file, chunk_len, all_characters))       
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, float(epoch) / n_epochs * 100, loss))
            print("%s \n"%evaluate(decoder,all_characters, 'Wh', 100))

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0


    plt.figure()
    plt.plot(all_losses)

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=50)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
args = argparser.parse_args()

train_net(args)





