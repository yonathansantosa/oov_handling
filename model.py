import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mimick(nn.Module):
    def __init__(self, embedding, char_emb_dim,emb_dim, hidden_size):
        super(mimick, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(char_emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size*2, 300),
            nn.ReLU(),
            nn.Linear(300, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
        )

    def forward(self, inputs):
        x_padded = nn.utils.rnn.pad_sequence(inputs[0], batch_first=True).to(device)
        x_packed_padded = nn.utils.rnn.pack_padded_sequence(self.embedding(x_padded), lengths=inputs[1], enforce_sorted=False, batch_first=True)
        _, (hidden_state, _) = self.lstm(x_packed_padded)
        out_cat = torch.cat((hidden_state[0, :, :], hidden_state[1, :, :]), 1)
        out = self.mlp(out_cat)

        return out

class mimick_cnn(nn.Module):
    def __init__(self, embedding, char_emb_dim=300, emb_dim=300, num_feature=100, features=[2,3,4,5,6,7], dropout=0.):
        '''
        CNN grams model for mimicking embedding
        
        Arguments:
        embedding = character embedding (nn.Embedding)
        char_emb_dim = character embedding dimension (int)
        num_feature = number of features for each n-grams (int)
        features = ordered list of used n-grams (list)
        dropout = dropout parameter (float)
        '''
        super(mimick_cnn, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.dropout = nn.Dropout(dropout)
        self.num_feature = num_feature

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, num_feature, (2, self.embedding.embedding_dim), bias=False),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, num_feature, (3, self.embedding.embedding_dim), bias=False),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, num_feature, (4, self.embedding.embedding_dim), bias=False),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1, num_feature, (5, self.embedding.embedding_dim), bias=False),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1, num_feature, (6, self.embedding.embedding_dim), bias=False),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1, num_feature, (7, self.embedding.embedding_dim), bias=False),
            nn.ReLU()
        )
        self.features = np.zeros(6)
        self.features[np.array(features)[:]-2] = 1
        self.bn1 = nn.BatchNorm1d(self.num_feature*6)
        self.mlp1 = nn.Sequential(nn.Linear(num_feature*6, 200), nn.Tanh(),)
        self.bn2 = nn.BatchNorm1d(200)
        self.log_sigma = nn.Sequential(nn.Linear(num_feature*6, emb_dim), nn.Tanh())
        self.mu = nn.Sequential(nn.Linear(num_feature*6, emb_dim), nn.Tanh())
        self.mlp2 = nn.Sequential(nn.Linear(200, emb_dim))
        self.t = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Tanh())

    def forward(self, inputs):
        '''
        Feedforwarding inputs to model
        
        inputs = input data (num_batch x channel x char_seq)
        output = output data (num_batch x channel x emb_dim)
        '''
        x = self.dropout(self.embedding(inputs).float())
        sizes = torch.Size((x.shape[0], self.num_feature, x.shape[3]))
        zeroes = torch.zeros(sizes, dtype=torch.float, device=x.device)
        
        x2 = self.conv2(x).squeeze(-1) if self.features[0] != 0 else zeroes
        x3 = self.conv3(x).squeeze(-1) if self.features[1] != 0 else zeroes
        x4 = self.conv4(x).squeeze(-1) if self.features[2] != 0 else zeroes
        x5 = self.conv5(x).squeeze(-1) if self.features[3] != 0 else zeroes
        x6 = self.conv6(x).squeeze(-1) if self.features[4] != 0 else zeroes
        x7 = self.conv7(x).squeeze(-1) if self.features[5] != 0 else zeroes

        x2_max = F.max_pool1d(x2, x2.shape[2]).squeeze(-1)
        x3_max = F.max_pool1d(x3, x3.shape[2]).squeeze(-1)
        x4_max = F.max_pool1d(x4, x4.shape[2]).squeeze(-1)
        x5_max = F.max_pool1d(x5, x5.shape[2]).squeeze(-1)
        x6_max = F.max_pool1d(x6, x6.shape[2]).squeeze(-1)
        x7_max = F.max_pool1d(x7, x7.shape[2]).squeeze(-1)
        
        maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=1)
        
        norm_maxpoolcat = self.bn1(maxpoolcat)
        out_cnn = self.mlp1(norm_maxpoolcat) 
        norm_out_cnn = self.bn2(out_cnn)
        out = self.mlp2(norm_out_cnn)

        return out


class mimick_cnn_ascii(nn.Module):
    def __init__(self, embedding, char_emb_dim=300, emb_dim=300, dropout=0.):
        '''
        Character Embedding model with binary representation
        
        init:
        -----
            CNN grams model for mimicking embedding
            
            Arguments:
            embedding = character embedding (nn.Embedding)
            char_emb_dim = character embedding dimension (int)
            emb_dim = word embedding dimension (int)
        '''
        super(mimick_cnn_ascii, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.dropout = nn.Dropout(.5)
        # self.num_feature = num_feature

        self.char_size = char_emb_dim
        kernel_sizes = [100, 50, 20, 10]

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, kernel_sizes[0], kernel_size=(2, self.char_size), padding=(1,0)),
            self.dropout,
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d((2,1), ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, kernel_sizes[1], (2, kernel_sizes[0]), padding=(1,0)),
            self.dropout,
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d((2,1), ceil_mode=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, kernel_sizes[2], (2, kernel_sizes[1]), padding=(1,0)),
            self.dropout,
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d((2,1), ceil_mode=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, kernel_sizes[3], (2, kernel_sizes[2]), padding=(1,0)),
            self.dropout,
            nn.ReLU()
        )
        
        self.FC = nn.Sequential(
            nn.Linear(2, 150),
            nn.ReLU(),
            nn.Linear(150, emb_dim)
        )

    def forward(self, inputs):
        '''
        Forward pass for the model

        input:
        ------
            x (torch.FloatTensor) : Inputs with the size of N x C x H x W
                N : number of data/minibatch size
                C : number of channel(s), must be size of 1
                H : height, must be equal with char_size
                W : width
            
        output:
        -------
            out (torch.FloatTensor) : output with the size of N x E
                N : number of data/minibatch size
                E : word embedding size
        '''
        x = self.dropout(self.embedding(inputs).float())

        if x.shape[3] != self.char_size:
            raise Exception('Character size of inputs and model is different')
        
        # N x 1 x emb_dim x L
        c1 = self.conv1(x)
        c1 = torch.reshape(c1, (c1.shape[0], 1, c1.shape[2], -1))
        p1 = self.pool1(c1)


        # N x 1 x 30 x L1: ceil((L-1)/2)
        c2 = self.conv2(p1)
        c2 = torch.reshape(c2, (c2.shape[0], 1, c2.shape[3], -1))
        p2 = self.pool2(c2)


        # N x 1 x 30 x L1: ceil(((L-1)/2) - 1)/2)
        c3 = self.conv3(p2)
        c3 = torch.reshape(c3, (c3.shape[0], 1, c3.shape[2], -1))
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        c4 = torch.reshape(c4, (c4.shape[0], 1, c4.shape[2], -1))
        p4 = F.max_pool2d(c4, kernel_size=(1, c4.shape[3]), ceil_mode=True)

        p4 = torch.reshape(p4, (p4.shape[0], -1))

        out = self.FC(p4)

        return out