import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F

class Char_embedding:
    def __init__(self, char_emb_dim=300, char_max_len=15, random=False, embed_type='random', device='cuda', freeze=False):
        super(Char_embedding, self).__init__()
        '''
        Initializing character embedding
        Parameter:
        emb_dim = (int) embedding dimension for character embedding
        ascii = mutually exclusive with random
        '''
        self.char_max_len = char_max_len
        self.embedding_type = embed_type
        self.embedding_folder = 'embeddings/char_embedding'
        if self.embedding_type=='random':
            torch.manual_seed(5)
            table = np.transpose(np.loadtxt(f'{self.embedding_folder}/glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
            self.weight_char = np.transpose(table[1:].astype(np.float))
            self.char = np.transpose(table[0])
            self.embed = nn.Embedding(len(self.char), char_emb_dim).to(device)
            self.embed.weight[1] = torch.zeros(char_emb_dim)
        elif self.embedding_type=='ascii':
            table = np.transpose(np.loadtxt(f'{self.embedding_folder}/ascii.embedding.txt', dtype=str, delimiter=' ', comments='##'))
            self.char = np.transpose(table[0])
            self.weight_char = np.transpose(table[1:].astype(np.float))

            self.weight_char = torch.from_numpy(self.weight_char).to(device)
            
            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=freeze)
            char_emb_dim = 8
        elif self.embedding_type=='onehot':
            chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.!@#$%^&*|()_+-=[]<>?/\\;\'"\t\n '
            self.char = [c for c in chars]
            self.char = ['<unk>', '<pad>', '<eow>', '<sow>'] + self.char
            ix = np.arange(len(self.char))
            self.char2ix = dict(zip(self.char,ix))
            self.char_size = len(self.char)
            self.weight_char = torch.transpose(nn.functional.one_hot(torch.arange(len(self.char), dtype=torch.long), num_classes = len(self.char)+1), 1,0)
            self.embed = nn.Embedding.from_pretrained(self.weight_char.type(torch.float)[:-1], freeze=True)
            char_emb_dim = self.char_size
        else:
            table = np.transpose(np.loadtxt(f'{self.embedding_folder}/glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
            self.char = np.transpose(table[0])
            self.weight_char = np.transpose(table[1:].astype(np.float))
            self.weight_char = self.weight_char[:,:char_emb_dim]

            self.weight_char = torch.from_numpy(self.weight_char).to(device)
            
            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=freeze)

        self.embed.padding_idx = 1
        self.char2idx = {}
        self.idx2char = {}
        self.char_emb_dim = char_emb_dim
        for i, c in enumerate(self.char):
            self.char2idx[c] = int(i)
            self.idx2char[i] = c

    def char_split(self, sentence, model_name='lstm', dropout=0.):
        '''
        Splitting character of a sentences then converting it
        into list of index

        Parameter:

        sentence = list of words
        '''
        char_data = []
        
        for word in sentence:
            c = list(word)
            c = ['<sow>'] + c +['<eow>']
            c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
            if model_name != 'lstm':
                c_idx = c_idx + [self.char2idx['<pad>']] * 7
            char_data += [torch.LongTensor(c_idx)]
        return char_data

    def char_sents_split(self, sentences, model_name='lstm', dropout=0.):
        '''
        Splitting character of a sentences then converting it
        into list of index

        Parameter:

        sentence = list of words
        '''
        sents_data = []
        for sentence in sentences:
            char_data = []
            for word in sentence:
                if word == '<pad>':
                    c_idx = torch.LongTensor([self.char2idx['<pad>']] * 7)
                else:
                    c = list(word)
                    c = ['<sow>'] + c +['<eow>']
                    c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                    if model_name != 'lstm':
                        c_idx = c_idx + [self.char2idx['<pad>']] * 7
                char_data += [torch.LongTensor(c_idx)]
            sents_data += char_data

        return sents_data

    def char2ix(self, c):
        if c in self.char2idx:
            return self.char2idx[c]
        else:
            return self.char2idx['<unk>']

    def ix2char(self, idx):
        return self.idx2char[idx]

    def idxs2word(self, idxs):
        return "".join([self.idx2char[idx] for idx in idxs])

    def word2idxs(self, word, model_name='lstm'):
        if word != '<pad>':
            c = list(word)
            c = ['<sow>'] + c +['<eow>']
            char_data = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
            if model_name != 'lstm':
                char_data = char_data + [self.char2idx['<pad>']] * 7
        else:
            char_data = [self.char2idx['<pad>']] * 7

        return torch.LongTensor(char_data)

    def clean_idxs2word(self, idxs):
        idxs = [i for i in idxs if i != 0 and i != 1 and i != 2 and i != 3]
        return "".join([self.idx2char[idx] for idx in idxs])

    def get_char_vectors(self, words):
        sentence = []
        for idxs in words:
            sentence += [self.char_embedding(idxs)]
            
        return torch.stack(sentence).permute(1, 0, 2)