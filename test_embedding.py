import torch
import torch.nn as nn
import os

import numpy as np

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse


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

    # result = (tensor1_dot_tensor2/divisor).data.cpu().numpy()
    result = (tensor1_dot_tensor2/divisor.clamp(min=1.e-09)).data.cpu()
    d, n = torch.sort(result, descending=True)
    n = n[:, :neighbor]
    d = d[:, :neighbor]
    return d, n


# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30, help='maximum iteration (default=1000)')
parser.add_argument('--run', default=1, help='starting epoch (default=1)')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true', help='whether to load model or not')
parser.add_argument('--lang', default='en', help='choose which language for word embedding')
parser.add_argument('--model', default='lstm', help='choose which mimick model')
parser.add_argument('--lr', default=0.1, help='learning rate')
parser.add_argument('--charlen', default=20, help='maximum length')
parser.add_argument('--charembdim', default=300)
parser.add_argument('--char_embed_type', default='random')
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--dropout', default=0)
parser.add_argument('--bsize', default=64)
parser.add_argument('--epoch', default=0)
parser.add_argument('--quiet', default=False, action='store_true')
parser.add_argument('--init_weight', default=False, action='store_true')
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--nesterov', default=False, action='store_true')
parser.add_argument('--loss_reduction', default=False, action='store_true')
parser.add_argument('--num_feature', default=50)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--momentum', default=0)
parser.add_argument('--multiplier', default=1)
parser.add_argument('--classif', default=200)
parser.add_argument('--neighbor', default=5)
parser.add_argument('--seed', default=128)
parser.add_argument('--words', nargs='+')
parser.add_argument('--oov_list', default=False, action='store_true')
parser.add_argument('--cnngrams', nargs='+')

args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'train_dropout/trained_model_{args.lang}_{args.model}_{args.embedding}_{args.seed}_{args.num_feature}'
if args.cnngrams != None and args.model == 'cnn': saved_model_path += '_' + ''.join(args.cnngrams)
logger_dir = f'{saved_model_path}/logs/run{args.run}/'
logger_val_dir = f'{saved_model_path}/logs/val-{args.run}/'

classif = int(args.classif)
multiplier = int(args.multiplier)

if not args.local: saved_model_path = cloud_dir + saved_model_path

if args.loss_fn == 'cosine':
    print('true')
# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
neighbor = int(args.neighbor)
dropout = int(args.dropout)
char_embed = Char_embedding(args.charembdim, args.charlen, embed_type=args.char_embed_type, random=True, device=device)
dataset = Word_embedding(lang=args.lang, embedding=args.embedding)
emb_dim = dataset.emb_dim

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

model.to(device)

model.load_state_dict(torch.load(f'{saved_model_path}/{args.model}.pth'))

model.eval()

words = 'MCT McNeally Vercellotti Secretive corssing flatfish compartmentalize pesky lawnmower developiong hurtling expectedly'.split()

# *Evaluating
if args.oov_list:
    f = open(f'oov_list_{args.embedding}.txt', 'r')
    words += f.read().split()
elif args.words != None:
    words = args.words


idxs = char_embed.char_split(words, model_name=args.model)
if args.model == 'lstm':
    idxs_len = torch.LongTensor([len(data) for data in idxs])
    inputs = (idxs, idxs_len)
else: 
    idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
    idxs = idxs.unsqueeze(1)
    inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
output = model.forward(inputs) # (batch x word_emb_dim)

cos_dist, nearest_neighbor = cosine_similarity(output.to(device), dataset.word_embedding.weight.to(device), neighbor)

for i, word in enumerate(words):
    if word not in dataset.stoi: print('OOV', end=' ')
    print(f'{word} & {dataset.idxs2sentence(nearest_neighbor[i])}\\\\')
    # if i % 2 == 1:
    #     print('\\\\')
    # else:
    #     print(' & ', end='')
    # print(f'{cos_dist[i][0].item():.4f} | {word} \t=>
    # {dataset.idxs2sentence(nearest_neighbor[i])}')
    