import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import argparse

def cosine_similarity(tensor):
    similarity = []
    t1 = tensor[:, 0, :]
    t2 = tensor[:, 1, :]

    return F.cosine_similarity(t1, t2)

class Word_similarity:
    def __init__(self, stoi=None, dataset='men'):
        self.file_name = None
        self.word_pairs = []
        self.scores = []
        self.stoi = stoi
        dataset_folder = 'dataset/word_similarity'
        dataset_file = {
            'card' : f'{dataset_folder}/Card-660.txt',
            'mc30': f'{dataset_folder}/MC-30.txt',
            'men': f'{dataset_folder}/MEN-TR-3k.txt',
            'mturk287': f'{dataset_folder}/MTurk-287.txt',
            'mturk771' : f'{dataset_folder}/MTurk-771.txt',
            'rg65' : f'{dataset_folder}/RG-65.txt',
            'rwstanford': f'{dataset_folder}/RW-STANFORD.txt',
            'simlex': f'{dataset_folder}/SimLex999.txt',
            'simverb': f'{dataset_folder}/SimVerb-3500.txt',
            'verb143': f'{dataset_folder}/verb143.txt',
            'wordsim': f'{dataset_folder}/wordsim353.txt',
            'yp130': f'{dataset_folder}/YP-130.txt'
        }

        self.file_name = dataset_file[dataset]
    
        with open(self.file_name, 'r') as f:
            for line in f:
                split = line[:-1].split('\t')
                w1 = split[0]
                w2 = split[1]
                score = float(split[2])
                self.word_pairs += [[w1, w2]]
                self.scores += [score]

        self.scores = np.array(self.scores)
    
    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        w1, w2 = self.word_pairs[index]
        emb_w1 = self.stoi[w1]
        emb_w2 = self.stoi[w2]
        return (torch.LongTensor([emb_w1, emb_w2]), torch.FloatTensor([self.scores[index]]))

    def get_pairs(self, index):
        return self.word_pairs[index]

    def get_scores(self):
        return torch.FloatTensor(self.scores)

    def remove_oov(self):
        remove_idx = []
        for i, (w1, w2) in enumerate(self.word_pairs):
            if w1 not in self.stoi or w2 not in self.stoi:
                remove_idx += [i]

        remove_idx = remove_idx[::-1]

        for idx in remove_idx:
            self.word_pairs.pop(idx)
            self.scores.pop(idx)


# *Argument parser
parser = argparse.ArgumentParser(
    description='Word similarity task for OOV handling'
)

parser.add_argument('--lang', default='en',
                    help='choose which language for word embedding')
parser.add_argument('--model', default='lstm',
                    help='choose which mimick model')
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--asc', default=False, action='store_true')
parser.add_argument('--charlen', default=20, help='maximum length')
parser.add_argument('--charembdim', default=300)
parser.add_argument('--num_feature', default=50)
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--save', default=False, action='store_true')
parser.add_argument('--cnngrams', nargs='+')
parser.add_argument('--trained_seed', default=64)

args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'train_dropout/trained_model_{args.lang}_{args.model}_{args.embedding}_{args.trained_seed}_{args.num_feature}'
if args.cnngrams != None and args.model == 'cnn': saved_model_path += '_' + ''.join(args.cnngrams)

if not args.local:
    saved_model_path = cloud_dir + saved_model_path

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
multiplier = int(args.multiplier)
classif = int(args.classif)

# *Hyperparameter
batch_size = 64
char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args.asc, random=True, device=device)
word_embedding = Word_embedding(lang=args.lang, embedding=args.embedding)
word_embedding.word_embedding.to(device)
emb_dim = word_embedding.emb_dim

#* Initializing model
if args.load:
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
            char_max_len=char_embed.char_max_len, 
            char_emb_dim=char_embed.char_emb_dim, 
            emb_dim=emb_dim,
            num_feature=int(args.num_feature),
            random=False, asc=args.asc, features=features,
            dropout=dropout)
    model.to(device)
    model.load_state_dict(torch.load(f'{saved_model_path}/{args.model}.pth'))
    model.eval()

#! later uncomment this
print('dataset,oov,invocab,rho,p')
dataset_names = ['card', 'mc30', 'men', 'mturk287', 'mturk771', 'rg65', 'rwstanford', 'simlex', 'simverb', 'verb143', 'wordsim', 'yp130']
for d_names in dataset_names:
    new_word = []
    oov = 0
    invocab = 0
    dataset = Word_similarity(stoi=word_embedding.stoi, dataset=d_names)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    word_pairs = set(np.array(dataset.word_pairs).flatten())

    for w in word_pairs:
        if w not in word_embedding.stoi:
            word_embedding.stoi[w] = len(word_embedding.stoi)
            word_embedding.itos += [w]
            if args.load:
                idxs = char_embed.word2idxs(w, args.model).unsqueeze(0).to(device).detach()
                if args.model == 'lstm':
                    idxs_len = torch.LongTensor([len(data) for data in idxs])
                    output = model.forward((idxs, idxs_len)).detach()
                else: 
                    idxs = nn.utils.rnn.pad_sequence(idxs, batch_first=True)
                    idxs = idxs.unsqueeze(1)
                    inputs = Variable(idxs).to(device) # (batch x channel x seq_len)
                    output = model.forward(inputs).detach()
                new_word += [output.cpu()]
            else:
                new_word += [torch.randn(emb_dim)]
            oov += 1
        else:
            invocab += 1

    if new_word != []:
        new_word = torch.stack(new_word).view(-1, emb_dim).to(device)
        word_embedding.word_embedding.weight.data = torch.cat((word_embedding.word_embedding.weight.data, new_word)).to(device)
        
    similarity = []
    if args.save: f = open(f'word_similarity_result/{d_names}_result.txt', 'w')
    for it, (X, _) in enumerate(data_loader):
        words = [word_embedding.idxs2sentence(w) for w in X]
        embeddings = word_embedding.word_embedding(X.to(device))
        cos_sim = cosine_similarity(embeddings)
        similarity.extend([cos_sim])
        if args.save:        
            for i, w in enumerate(words):
                f.write(f'{w} {float(cos_sim[i]):.4f}\n')
    if args.save: f.close() 

    r, p = spearmanr(torch.cat(similarity).detach().cpu().numpy(), np.array(dataset.scores))
    
    print(f'{d_names},{oov},{invocab},{r:.8f},{p:.8f}')