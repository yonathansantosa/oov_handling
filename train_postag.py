import nltk
import numpy as np
from nltk.corpus import brown
from tagset.tagset import Postag, Postagger_adaptive

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler, DataLoader

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse
from tqdm import trange, tqdm
import os
from logger import Logger
import shutil
from distutils.dir_util import copy_tree

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)

def sort_idxs(idxs, argsort):
    new_idxs = [idxs[i] for i in argsort]
    return new_idxs

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30, help='maximum iteration (default=1000)')
parser.add_argument('--run', default=0, help='starting epoch (default=1000)')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true', help='whether to load model or not')
parser.add_argument('--test', default=False, action='store_true', help='whether to only test the model or not')
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
parser.add_argument('--oov_random', default=False, action='store_true')
parser.add_argument('--num_feature', default=100)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--momentum', default=0)
parser.add_argument('--neighbor', default=5)
parser.add_argument('--seq_len', default=5)
parser.add_argument('--seed', default=64)
parser.add_argument('--trained_seed', default=64)
parser.add_argument('--tagset', default='brown')
parser.add_argument('--freeze', default=False, action='store_true')
parser.add_argument('--train_embed', default=False, action='store_true')
parser.add_argument('--cnngrams', nargs='+')

args = parser.parse_args()

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'train_dropout/trained_model_{args.lang}_{args.model}_{args.embedding}_{args.trained_seed}_{args.num_feature}'
if args.cnngrams != None and args.model == 'cnn': saved_model_path += '_' + ''.join(args.cnngrams)
saved_postag_path = 'train_dropout/trained_model_%s_%s_%s_postag' % (args.lang, args.model, args.embedding)
logger_dir = '%s/logs/run%s/' % (saved_postag_path, args.run)
logger_val_dir = '%s/logs/val-run%s/' % (saved_postag_path, args.run)
logger_val_cosine_dir = '%s/logs/val-cosine-run%s/' % (saved_postag_path, args.run)

if not args.local:
    saved_model_path = cloud_dir + saved_model_path
    saved_postag_path = cloud_dir + saved_postag_path

logger = Logger(logger_dir)
logger_val = Logger(logger_val_dir)
#endregion

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
random_seed = int(args.seed)
shuffle_dataset = args.shuffle
validation_split = .8
neighbor = int(args.neighbor)
seq_len = int(args.seq_len)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
tag_set = args.tagset

# *Hyperparameter
batch_size = int(args.bsize)
val_batch_size = 64
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
weight_decay = float(args.weight_decay)
momentum = float(args.momentum)
dropout = float(args.dropout)

# model and embedding init
char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args.asc, random=True, device=device)

#* Initializing model
word_embedding = Word_embedding(lang=args.lang, embedding=args.embedding, freeze=args.train_embed, sparse=False)
emb_dim = word_embedding.emb_dim

if not args.oov_random:
    if args.model == 'lstm':
        model = mimick(
            embedding = char_embed.embed, 
            char_emb_dim = char_embed.char_emb_dim, 
            emb_dim = word_embedding.emb_dim, 
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

    model.to(device)
    model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))

    model.eval()

dataset = Postag(word_embedding)

#* Creating PT data samplers and loaders:
original_vocab_len = len(word_embedding)
if not args.quiet: print('total word = %d' % len(word_embedding))
new_word = []
oov = 0
invocab = 0
tagged_words = set([word for word, _ in dataset.tagged_words])
for word in tagged_words:
    if word not in word_embedding.stoi:
        word_embedding.stoi[word] = len(word_embedding.stoi)
        word_embedding.itos += [word]
        if args.oov_random:
            new_word += [torch.normal(torch.zeros(emb_dim, dtype=torch.float, device=device), std=3.0, out=None)]
        elif not args.freeze:
            new_word += [torch.zeros(emb_dim, dtype=torch.float, device=device)]
        else:
            idxs = char_embed.word2idxs(word, args.model).unsqueeze(0).to(device).detach()
            if args.model == 'lstm':
                idxs_len = torch.LongTensor([len(data) for data in idxs])
                generated_embeddings = model.forward((idxs, idxs_len)).detach()
            else: 
                idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
                idxs = idxs.unsqueeze(1)
                inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
                generated_embeddings = model.forward(inputs).detach()
            # output = model.forward(inputs).detach()
            new_word += [generated_embeddings]
        oov += 1
    else:
        invocab += 1

if not args.quiet: 
    print('oov = %d' % oov)
    print('invocab = %d' % invocab)

if args.oov_random:
    new_word += [torch.normal(torch.zeros(emb_dim, dtype=torch.float, device=device), std=3.0, out=None)]
elif not args.freeze:
    new_word += [torch.zeros(emb_dim, dtype=torch.float, device=device)]
else:
    idxs = char_embed.word2idxs('<pad>').unsqueeze(0).to(device).detach()
    if args.model == 'lstm':
        idxs_len = torch.LongTensor([len(data) for data in idxs])
        generated_embeddings = model.forward((idxs, idxs_len)).detach()
    else: 
        idxs = idxs.unsqueeze(1)
        inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
        generated_embeddings = model.forward(inputs).detach()                 
    new_word += [generated_embeddings]
    
new_word = torch.stack(new_word).squeeze()
        
word_embedding.stoi['<pad>'] = len(word_embedding.stoi)
word_embedding.itos += ['<pad>']
with torch.no_grad():
    word_embedding.word_embedding.weight.set_(torch.cat((word_embedding.word_embedding.weight.data, new_word)))
    
# train val split and loader
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

train_indices, val_indices = indices[:split], indices[split:]

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                sampler=valid_sampler)

postagger = Postagger_adaptive(seq_len, emb_dim, 20, len(dataset.tagset)).to(device)

if args.load:
    postagger.load_state_dict(torch.load('%s/postagger.pth' % (saved_postag_path)))
if args.load and args.train_embed:
    word_embedding.word_embedding.load_state_dict(torch.load('%s/embedding.pth' % (saved_postag_path)))
    
if not args.freeze and not args.oov_random:
    params = list(postagger.parameters()) + list(model.parameters())
    if args.train_embed: params += list(word_embedding.word_embedding.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=args.nesterov)
else:
    optimizer = optim.SGD(list(postagger.parameters()) + list(word_embedding.word_embedding.parameters()), lr=learning_rate, momentum=momentum, nesterov=args.nesterov)
criterion = nn.NLLLoss()

step = 0

#* Training
if not args.test:
    for epoch in trange(int(args.epoch), max_epoch, total=max_epoch, initial=int(args.epoch)):
        loss_item = 0.
        postagger.train()
        
        if not (args.freeze or args.oov_random): model.train()
        if args.train_embed: word_embedding.word_embedding.train()
        for it, (X, y) in enumerate(train_loader):
            postagger.zero_grad()
            if args.freeze or args.oov_random:
                inputs = Variable(X).to(device)
                embeddings = word_embedding.word_embedding(inputs)
            else:
                words = [word_embedding.idxs2words(x) for x in X]
                idxs = char_embed.char_sents_split(words, args.model)
                if args.model == 'lstm':
                    idxs_len = torch.LongTensor([len(data) for data in idxs])
                    generated_embeddings = model.forward((idxs, idxs_len)).view(X.size(0),-1,emb_dim)
                else: 
                    idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
                    idxs = idxs.unsqueeze(1)
                    inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
                    generated_embeddings = model.forward(inputs).view(X.size(0),-1,emb_dim)
                
                mask = (X < original_vocab_len).type(torch.FloatTensor).unsqueeze(2).to(device)
                words = Variable(X).to(device)
                pretrained_embeddings = word_embedding.word_embedding(words)
                embeddings = mask * pretrained_embeddings + (1-mask) * generated_embeddings

            target = Variable(y).to(device)
            output, loss = postagger.forward(embeddings, target)
            
            # Tensorboard
            loss_item = loss.item()
            info = {
                'loss-Train-%s-postag-run%s' % (args.model, args.run) : loss_item,
            }
            step += 1
            if args.run != 0:
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if not args.local:
            copy_tree(logger_dir, cloud_dir+logger_dir)
            
        if not args.quiet: tqdm.write('%d | %.4f ' % (epoch, loss_item))

        #* Validation
        postagger.eval()
        if not (args.freeze or args.oov_random): model.eval()
        if args.train_embed: word_embedding.word_embedding.eval()
        validation_loss = 0.
        accuracy = 0.
        for it, (X, y) in enumerate(validation_loader):
            if args.freeze or args.oov_random:
                inputs = Variable(X).to(device)
                embeddings = word_embedding.word_embedding(inputs)
            else:
                words = [word_embedding.idxs2words(x) for x in X]
                idxs = char_embed.char_sents_split(words, args.model)
                if args.model == 'lstm':
                    idxs_len = torch.LongTensor([len(data) for data in idxs])
                    generated_embeddings = model.forward((idxs, idxs_len)).view(X.size(0),-1,emb_dim)
                else: 
                    idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
                    idxs = idxs.unsqueeze(1)
                    inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
                    generated_embeddings = model.forward(inputs).view(X.size(0),-1,emb_dim)
                mask = (X < original_vocab_len).type(torch.FloatTensor).unsqueeze(2).to(device)
                pretrained_embeddings = word_embedding.word_embedding(Variable(X).to(device))
                embeddings = mask * pretrained_embeddings + (1-mask) * generated_embeddings
                
            target = Variable(y).to(device)
            output, validation_loss = postagger.validation(embeddings, target)
            output_tag = output.view(X.shape[0], seq_len)
            accuracy += float((output_tag == target).sum())

            if not args.quiet:
                if it == 0:
                    tag = output_tag[0]
                    for i in range(len(X[0])):
                        word_idx = X[0][i].cpu().numpy()
                        word = word_embedding.idx2word(word_idx)
                        tg = dataset.tagset.idx2tag(int(tag[i].cpu()))
                        tgt = dataset.tagset.idx2tag(int(y[0][i]))
                        tqdm.write('(%s, %s) => %s' % (word, tgt, tg))
        accuracy /= (seq_len*len(val_indices))
        if not args.quiet: tqdm.write('accuracy = %.4f' % accuracy)

        info_val = {
            'loss-Train-%s-postag-run%s' % (args.model, args.run) : validation_loss
        }
        if args.run != 0:
            for tag, value in info_val.items():
                logger_val.scalar_summary(tag, validation_loss, step)

        if not args.quiet: tqdm.write('val_loss %.4f ' % validation_loss)
        
        postagger.train()

    if args.save: 
        torch.save(postagger.state_dict(), f'{saved_postag_path}/postagger.pth')
        torch.save(word_embedding.word_embedding.state_dict(), f'{saved_postag_path}/embedding.pth')
    postagger.eval()
if not (args.freeze or args.oov_random): model.eval()

accuracy = 0.
for it, (X, y) in enumerate(validation_loader):
    if args.freeze or args.oov_random:
        inputs = Variable(X).to(device)
        embeddings = word_embedding.word_embedding(inputs)
    else:
        words = [word_embedding.idxs2words(x) for x in X]
        idxs = char_embed.char_sents_split(words, args.model)
        if args.model == 'lstm':
            idxs_len = torch.LongTensor([len(data) for data in idxs])
            generated_embeddings = model.forward((idxs, idxs_len)).view(X.size(0),-1,emb_dim)
        else: 
            idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
            idxs = idxs.unsqueeze(1)
            inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
            generated_embeddings = model.forward(inputs).view(X.size(0),-1,emb_dim)
        mask = (X < original_vocab_len).type(torch.FloatTensor).unsqueeze(2).to(device)
        pretrained_embeddings = word_embedding.word_embedding(Variable(X).to(device))
        embeddings = mask * pretrained_embeddings + (1-mask) * generated_embeddings
    target = Variable(y).to(device)
    output, _ = postagger.validation(embeddings, target)
    output_tag = output.view(X.shape[0], seq_len)
    accuracy += float((output_tag == target).sum())
    if it <= 3 and not args.quiet:
        tag = output_tag[0]
        for i in range(len(X[0])):
            word_idx = X[0][i].cpu().numpy()
            word = word_embedding.idx2word(word_idx)
            tg = dataset.tagset.idx2tag(int(tag[i].cpu()))
            tgt = dataset.tagset.idx2tag(int(y[0][i]))
            tqdm.write('(%s, %s) => %s' % (word, tgt, tg))
        tqdm.write('\n')
accuracy /= (seq_len*len(val_indices))
print('accuracy = %.4f' % accuracy)
    