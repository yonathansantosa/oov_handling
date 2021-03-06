import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, gradcheck
from torch.utils.data import SubsetRandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import math

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse
from tqdm import trange, tqdm
import os
# from logger import Logger
import shutil
from distutils.dir_util import copy_tree
import pickle

def cosine_similarity(tensor1, tensor2, neighbor=5):
    '''
    Calculating cosine similarity for each vector elements of
    tensor 1 with each vector elements of tensor 2

    Input:

    tensor1 = (torch.FloatTensor) with size N x D
    tensor2 = (torch.FloatTensor) with size M x D
    neighbor = (int) number of closest vector to be returned

    Output:

    (distance, neighbor)
    '''
    tensor1_norm = torch.norm(tensor1, 2, 1)
    tensor2_norm = torch.norm(tensor2, 2, 1)
    tensor1_dot_tensor2 = torch.mm(tensor2, torch.t(tensor1)).t()

    divisor = [t * tensor2_norm for t in tensor1_norm]

    divisor = torch.stack(divisor)

    result = (tensor1_dot_tensor2/divisor.clamp(min=1.e-09)).data.cpu()
    d, n = torch.sort(result, descending=True)
    n = n[:, :neighbor]
    d = d[:, :neighbor]
    return d, n

def init_weights(m):
    '''
    Initializing weights and biases with 0.01
    '''
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data = torch.normal(0, 0.05, m.weight.data.size())
        m.bias.data = torch.normal(0, 0.05, m.bias.data.size())

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30, type=int, help='maximum iteration (default=1000)')
parser.add_argument('--run', default=1, type=int, help='starting epoch (default=1)')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true', help='whether to load model or not')
parser.add_argument('--lang', default='en', help='choose which language for word embedding')
parser.add_argument('--model', default='lstm', help='choose which mimick model')
parser.add_argument('--lr', default=0.1, help='learning rate')
parser.add_argument('--charlen', default=20, help='maximum length', type=int)
parser.add_argument('--charembdim', default=300, type=int)
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--bsize', default=64, type=int)
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--char_embed_type', default='random')
parser.add_argument('--quiet', default=False, action='store_true')
parser.add_argument('--init_weight', default=False, action='store_true')
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--nesterov', default=False, action='store_true')
parser.add_argument('--loss_reduction', default=False, action='store_true')
parser.add_argument('--num_feature', default=50, type=int)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--multiplier', default=1, type=float)
parser.add_argument('--neighbor', default=5, type=int)
parser.add_argument('--seed', default=64, type=int)
parser.add_argument('--cnngrams', nargs='+')
parser.add_argument('--test', default=False, action='store_true')

args = parser.parse_args()

#* Create a failsafe in case of wrong numbering

if args.run != 1:
    args.load = True

# Folder naming based on train location
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'train_dropout/trained_model_{args.lang}_{args.model}_{args.embedding}_{args.seed}_{args.num_feature}'
if args.cnngrams != None and args.model == 'cnn': saved_model_path += '_' + ''.join(args.cnngrams)
logger_dir = f'{saved_model_path}/logs/run{args.run}/'
logger_val_dir = f'{saved_model_path}/logs/val-{args.run}/'
if not args.local: saved_model_path = cloud_dir + saved_model_path
# logger = Logger(logger_dir)
# logger_val = Logger(logger_val_dir)
logger = SummaryWriter(logger_dir)
logger_val = SummaryWriter(logger_val_dir)

# Device configuration
torch.cuda.current_device()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shuffle_dataset = args.shuffle
validation_split = .8
neighbor = args.neighbor
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Hyperparameter
batch_size = int(args.bsize)
val_batch_size = 64
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
weight_decay = float(args.weight_decay)
momentum = float(args.momentum)
multiplier = float(args.multiplier)
dropout = float(args.dropout)

# Loading word and char embedding
char_embed = Char_embedding(args.charembdim, args.charlen, embed_type=args.char_embed_type, random=True, device=device)
dataset = Word_embedding(lang=args.lang, embedding=args.embedding, device=torch.device('cpu'))
emb_dim = dataset.emb_dim

# Creating dataset split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)

# Creating PT data samplers and loaders:
train_indices, val_indices = indices[:split], indices[split:]
np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SequentialSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=valid_sampler)

# Initializing model
if args.model == 'lstm':
    model = mimick(
        embedding = char_embed.embed, 
        char_emb_dim = char_embed.char_emb_dim, 
        emb_dim = dataset.emb_dim, 
        hidden_size = int(args.num_feature))
elif args.model == 'cnn':
    if args.cnngrams != None:
        features = [int(g) for g in args.cnngrams]
    else:
        features = list(range(2,8))
    model = mimick_cnn(
        embedding=char_embed.embed,
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        features=features,
        dropout=dropout)
elif args.model == 'cnn_ascii':
    model = mimick_cnn_ascii(embedding=char_embed.embed,
    char_emb_dim=char_embed.char_emb_dim,
    emb_dim=dataset.emb_dim,
    dropout=dropout,
    num_feature=int(args.num_feature))

if args.init_weight: model.apply(init_weights)
model.to(device)

if args.load or args.test:
    model.load_state_dict(torch.load(f'{saved_model_path}/{args.model}.pth'))
    
elif not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
        
word_embedding = dataset.embedding_vectors.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=args.nesterov)
criterion = nn.MSELoss()
step = 0

# Training
if not args.test:
    for epoch in trange(int(args.epoch), max_epoch, total=max_epoch, initial=int(args.epoch)):
        for it, (X, y) in enumerate(train_loader):
            model.zero_grad()
            words = dataset.idxs2words(X)
            idxs = char_embed.char_split(words, args.model)
            if args.model == 'lstm':
                idxs_len = torch.LongTensor([len(data) for data in idxs])
                inputs = (idxs, idxs_len)
            else: 
                idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
                idxs = idxs.unsqueeze(1)
                inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
            target = Variable(y*multiplier).to(device) # (batch x word_emb_dim)
            output = model.forward(inputs) # (batch x word_emb_dim)
            loss = criterion(output, target)

            # Tensorboard saving
            loss_item = loss.item() if not args.loss_reduction else loss.mean().item()
            info = {
                f'loss-Train-{args.model}-run{args.run}' : loss_item,
            }
            step += 1
            for tag, value in info.items():
                # logger.scalar_summary(tag, value, step)
                logger.add_scalar(tag, value, step)
            
            # Backpropagation
            if not args.loss_reduction:
                loss.backward()
            else:
                loss = loss.mean(0)
                for i in range(len(loss)-1):
                    loss[i].backward(retain_graph=True)

                loss[len(loss)-1].backward()

            optimizer.step()
            optimizer.zero_grad()

            # sanity check
            if not args.quiet:
                if it % int(dataset_size/(batch_size*5)) == 0:
                    tqdm.write(f'loss = {loss.mean():.4f}')
                    model.eval()
                    random_input = np.random.randint(len(X))
                    words = dataset.idx2word(X[random_input]) # list of words  
                    distance, nearest_neighbor = cosine_similarity(output[random_input].detach().unsqueeze(0), word_embedding, neighbor=neighbor)
                    loss_dist = torch.dist(output[random_input], target[random_input]*multiplier)
                    tqdm.write(f'{step} {loss_dist.item():.4f} | {words} \t=> {dataset.idxs2sentence(nearest_neighbor[0])}')
                    model.train()
                    tqdm.write('')
        
        model.eval()

        # Saving trained model
        if not args.local: copy_tree(logger_dir, cloud_dir+logger_dir)
        torch.save(model.state_dict(), f'{saved_model_path}/{args.model}.pth')

        mse_loss = 0.
        for it, (X, target) in enumerate(validation_loader):
            words = dataset.idxs2words(X)
            idxs = char_embed.char_split(words, args.model)
            if args.model == 'lstm':
                idxs_len = torch.LongTensor([len(data) for data in idxs])
                inputs = (idxs, idxs_len)
            else: 
                idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
                idxs = idxs.unsqueeze(1)
                inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
            target = target.to(device) # (batch x word_emb_dim)
            output = model.forward(inputs) # (batch x word_emb_dim)
            mse_loss += ((output-target)**2 / ((dataset_size-split)*emb_dim)).sum().item()
            
            # Printing some output
            if not args.quiet:
                if it < 1:
                    distance, nearest_neighbor = cosine_similarity(output, word_embedding, neighbor=neighbor)
                    for i, word in enumerate(X):
                        if i >= 1: break
                        loss_dist = torch.dist(output[i], target[i])
                        tqdm.write(f'{loss_dist.item():.4f} | {dataset.idx2word(word)} \t=> {dataset.idxs2sentence(nearest_neighbor[i])}')

        total_val_loss = mse_loss

        if not args.quiet: print(f'total loss = {total_val_loss:.8f}')
        info_val = {
            f'loss-Train-{args.model}-run{args.run}' : total_val_loss
        }

        # Logging graph
        for tag, value in info_val.items():
            # logger_val.scalar_summary(tag, value, step)
            logger_val.add_scalar(tag, value, step)
        model.train()

        # if not args.local:
        #     copy_tree(logger_val_dir, cloud_dir+logger_val_dir)

    # Saving trained embedding as txt
    f = open(f'{saved_model_path}/trained_embedding_{args.model}.txt', 'w')
    logger.flush()
    # logger_val.flush()

# Printing some prediction on train and val set
# for it, (X, y) in enumerate(train_loader):
#     model.zero_grad()
#     words = dataset.idxs2words(X)
#     idxs = char_embed.char_split(words, args.model)
#     if args.model == 'lstm':
#         idxs_len = torch.LongTensor([len(data) for data in idxs])
#         inputs = (idxs, idxs_len)
#     else: 
#         idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
#         idxs = idxs.unsqueeze(1)
#         inputs = Variable(idxs).to(device) # (batch x channel x seq_len)

#     output = model.forward(inputs) # (batch x word_emb_dim)
    # if not args.test:
    #     for w, e in zip(words,output):
    #         try:
    #             f.write(f'{w} {" ".join(map(str,[weight for weight in e.data.cpu().tolist()]))}\n')
    #         except UnicodeEncodeError:
    #             pass

mse_loss = 0.
for it, (X, target) in enumerate(validation_loader):
    words = dataset.idxs2words(X)
    idxs = char_embed.char_split(words, args.model)
    if args.model == 'lstm':
        idxs_len = torch.LongTensor([len(data) for data in idxs])
        inputs = (idxs, idxs_len)
    else: 
        idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
        idxs = idxs.unsqueeze(1)
        inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
    target = target.to(device) # (batch x word_emb_dim)

    model.zero_grad()

    output = model.forward(inputs) # (batch x word_emb_dim)
    # if not args.test:
    #     for w, e in zip(words,output):
    #         try:
    #             f.write(f'{w} {" ".join(map(str,[weight for weight in e.data.cpu().tolist()]))}\n')
    #         except UnicodeEncodeError:
    #             pass
    mse_loss += ((output-target)**2 / ((dataset_size-split)*emb_dim)).sum()
    distance, nearest_neighbor = cosine_similarity(output, word_embedding, neighbor=neighbor)
    if it < 1 and not args.test:
        for i, word in enumerate(X):
            loss_dist = torch.dist(output[i], target[i])
            print(f'{loss_dist.item():.4f} | {dataset.idx2word(word)} \t=> {dataset.idxs2sentence(nearest_neighbor[i])}')
    else:
        for i, word in enumerate(X):
            loss_dist = torch.dist(output[i], target[i])
            print(f'{loss_dist.item():.4f} | {dataset.idx2word(word)} \t=> {dataset.idxs2sentence(nearest_neighbor[i])}')
print(f'loss = {mse_loss.item():.4f}')