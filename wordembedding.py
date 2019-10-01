import numpy as np
import torch
import torch.nn as nn

from polyglot.mapping import Embedding

from torchtext.vocab import Vectors, GloVe

class Word_embedding:
    def __init__(self, emb_dim=300, lang='en', embedding='polyglot', freeze=True, sparse=True, device=torch.device('cuda')):
        '''
        Initializing word embedding
        Parameter:
        emb_dim = (int) embedding dimension for word embedding
        '''
        if embedding == 'glove':
            # *GloVE
            glove =  Vectors(f'word_embedding/glove.6B.{emb_dim}d.txt', cache='./embeddings/')
            self.embedding_vectors = glove.vectors
            self.stoi = glove.stoi
            self.itos = glove.itos
        elif embedding == 'word2vec':
            # *word2vec
            word2vec = Vectors('word_embedding/GoogleNews-vectors-negative300.bin.gz.txt', cache='./embeddings/')
            self.embedding_vectors = word2vec.vectors
            self.stoi = word2vec.stoi
            self.itos = word2vec.itos
        elif embedding == 'polyglot':
            # *Polyglot
            polyglot_emb = Embedding.load('./embeddings/word_embedding/embeddings_pkl.tar.bz2')
            self.embedding_vectors = torch.from_numpy(polyglot_emb.vectors)
            self.stoi = polyglot_emb.vocabulary.word_id
            self.itos = [polyglot_emb.vocabulary.id_word[i] for i in range(len(polyglot_emb.vocabulary.id_word))]
        elif embedding == 'dict2vec':
            dict2vec = Vectors('word_embedding/dict2vec-vectors-dim100.vec', cache='./embeddings/')
            self.embedding_vectors = dict2vec.vectors
            self.stoi = dict2vec.stoi
            self.itos = dict2vec.itos
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors, freeze=(not freeze), sparse=sparse).to(device)
        self.emb_dim = self.embedding_vectors.size(1)
        
    def __getitem__(self, index):
        return (torch.tensor([index], dtype=torch.long), self.word_embedding(torch.tensor([index])).squeeze())

    def __len__(self):
        return len(self.itos)
    
    def update_weight(self, weight, freeze=True):
        new_emb = Vectors(weight)
        self.embedding_vectors = new_emb.vectors
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors, freeze=freeze, sparse=True)
        self.emb_dim = self.embedding_vectors.size(1)
        self.stoi = new_emb.stoi
        self.itos = new_emb.itos

    def word2idx(self, c):
        return self.stoi[c]

    def idx2word(self, idx):
        return self.itos[int(idx)]

    def idxs2sentence(self, idxs, separator=' '):
        return separator.join([self.itos[int(i)] for i in idxs])

    def sentence2idxs(self, sentence):
        word = sentence.split()
        return [self.stoi[w] for w in word]

    def idxs2words(self, idxs):
        '''
        Return tensor of indexes as a sentence
        
        Input:
        idxs = (torch.LongTensor) 1D tensor contains indexes
        '''
        sentence = [self.itos[int(idx)] for idx in idxs]
        return sentence

    def get_word_vectors(self):
        return self.word_embedding