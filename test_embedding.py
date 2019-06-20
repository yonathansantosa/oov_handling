import torch
import torch.nn as nn

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
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--dropout', default=0)
parser.add_argument('--bsize', default=64)
parser.add_argument('--epoch', default=0)
parser.add_argument('--asc', default=False, action='store_true')
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



args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'train_dropout/trained_model_{args.lang}_{args.model}_{args.embedding}_{args.seed}'
logger_dir = f'{saved_model_path}/logs/run{args.run}/'
logger_val_dir = f'{saved_model_path}/logs/val-{args.run}/'

classif = int(args.classif)
multiplier = int(args.multiplier)
args.local = True
if not args.local: saved_model_path = cloud_dir + saved_model_path

if args.loss_fn == 'cosine':
    print('true')
# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
neighbor = int(args.neighbor)

char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args.asc, random=True, device=device)
dataset = Word_embedding(lang=args.lang, embedding=args.embedding)
emb_dim = dataset.emb_dim

#* Initializing model
if args.model == 'lstm':
    model = mimick(char_embed.embed, char_embed.char_emb_dim, dataset.emb_dim, int(args.num_feature))
elif args.model == 'cnn2':
    model = mimick_cnn2(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature), 
        random=False, asc=args.asc)
elif args.model == 'cnn':
    model = mimick_cnn(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature), 
        random=False, asc=args.asc)
elif args.model == 'cnn3':
    model = mimick_cnn3(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        mtp=multiplier, 
        random=False, asc=args.asc)
elif args.model == 'cnn4':
    model = mimick_cnn4(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        classif=classif,
        random=False, asc=args.asc)
else:
    model = None

model.to(device)

model.load_state_dict(torch.load(f'{saved_model_path}/{args.model}.pth'))

model.eval()

# *Evaluating
if args.words == None:
    words = 'MCT McNeally Vercellotti Secretive corssing flatfish compartmentalize pesky lawnmower developiong hurtling expectedly'.split()
else:
    words = 'MCT McNeally Vercellotti Secretive corssing flatfish compartmentalize pesky lawnmower developiong hurtling expectedly'.split() + args.words

idxs = char_embed.char_split(words).to(device)
if args.model != 'lstm': idxs = idxs.unsqueeze(1)
inputs = Variable(idxs) # (length x batch x char_emb_dim)
output = model.forward(inputs) # (batch x word_emb_dim)

cos_dist, nearest_neighbor = cosine_similarity(output.to(device), dataset.word_embedding.weight.to(device), neighbor)

for i, word in enumerate(words):
    print(f'{cos_dist[i][0].item():.4f} | {word} \t=> {dataset.idxs2sentence(nearest_neighbor[i])}')