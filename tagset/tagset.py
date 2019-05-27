import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(0)

class Tagset:
    def __init__(self, tagset='brown'):
        self.itot = {}
        self.toti = {}
        with open (f'tagset/{tagset}.txt', "r") as myfile:
            data=myfile.readlines()
            sent = "".join([d for d in data])
            processed = re.findall(r"(.*):", sent)
            for i, tag in enumerate(processed):
                self.toti[tag] = i

            for i, tag in enumerate(processed):
                self.itot[i] = tag

    def __len__(self):
        return len(self.itot)
        
    def idx2tag(self, idx):
        return self.itot[idx]

    def tag2idx(self, tag):
        return self.toti[tag]


class Postag:
    def __init__(self, word_embed, corpus='brown', tagset='brown', device='cuda'):
        if corpus == 'brown':
            from nltk.corpus import brown as corpus
        self.word_embed = word_embed
        self.tagged_words = corpus.tagged_words(tagset=tagset)
        self.tagged_sents = corpus.tagged_sents(tagset=tagset)
        self.tagset = Tagset(tagset=tagset)
        new_itot = {}
        new_toti = {}
        self.count_bin = torch.zeros(len(self.tagset))
        self.idxs = torch.zeros(1)
        self.device = device

        for word, tag in self.tagged_words:
            if tag in self.tagset.toti:
                self.count_bin[self.tagset.tag2idx(tag)] += 1
            else:
                self.count_bin[self.tagset.tag2idx('UNK')] += 1
        
        _, self.idxs = torch.sort(self.count_bin, descending=True)

        for it, i in enumerate(self.idxs):
            new_itot[it] = self.tagset.itot[int(i)]
            new_toti[new_itot[it]] = it

        self.tagset.toti = new_toti
        self.tagset.itot = new_itot


    def __len__(self):
        return len(self.tagged_sents)


    def __getitem__(self, index):
        length = len(self.tagged_sents[index])
        word = []
        tag = []
        
        if length-5 <= 0:
            for i in range(length):
                w, t = self.tagged_sents[index][i]
                word += [self.word_embed.word2idx(w)]
                
                if t in self.tagset.toti:
                    tag_id = self.tagset.tag2idx(t)
                else:
                    tag_id = self.tagset.tag2idx('UNK')
                    
                tag += [tag_id]

            for i in range(length, 5):
                word += [self.word_embed.word2idx('<pad>')]
                tag_id = self.tagset.tag2idx('UNK')

                tag += [tag_id]

        else:
            start_index = np.random.randint(0, length-5)
            for i in range(start_index, start_index+5):
                w, t = self.tagged_sents[index][i]
                if t in self.tagset.toti:
                    tag_id = self.tagset.tag2idx(t)
                else:
                    tag_id = self.tagset.tag2idx('UNK')
                word += [self.word_embed.word2idx(w)]
                tag += [tag_id]

        return (torch.LongTensor(word), torch.LongTensor(tag))

class Postag_word:
    def __init__(self, word_embed, char_embed, corpus='brown', tagset='brown'):
        if corpus == 'brown':
            from nltk.corpus import brown as corpus
        self.char_embed = char_embed
        self.tagged_words = corpus.tagged_words(tagset=tagset)
        self.tagged_sents = corpus.tagged_sents(tagset=tagset)
        self.tagset = Tagset(tagset=tagset)
        new_itot = {}
        new_toti = {}
        self.word_embed = word_embed
        self.count_bin = torch.zeros(len(self.tagset))
        self.idxs = torch.zeros(1)

        for _, tag in self.tagged_words:
            if tag in self.tagset.toti:
                self.count_bin[self.tagset.tag2idx(tag)] += 1
            else:
                self.count_bin[self.tagset.tag2idx('UNK')] += 1
        
        _, self.idxs = torch.sort(self.count_bin, descending=True)

        for it, i in enumerate(self.idxs):
            new_itot[it] = self.tagset.itot[int(i)]
            new_toti[new_itot[it]] = it

        self.tagset.toti = new_toti
        self.tagset.itot = new_itot


    def __len__(self):
        return len(self.tagged_sents)


    def __getitem__(self, index):
        word, tag = self.tagged_words[index]

        w_c_idx = self.char_embed.word2idxs(word)
        if tag in self.tagset.toti:
            tag_id = self.tagset.tag2idx(tag)
        else:
            tag_id = self.tagset.tag2idx('UNK')
        
        try:
            w_idx = self.word_embed.stoi[word]
        except:
            pass

        return (torch.LongTensor(w_idx), torch.LongTensor(w_c_idx), torch.LongTensor(tag_id))

class Postagger(nn.Module):
    def __init__(self, seq_length, emb_dim, hidden_size, output_size):
        super(Postagger, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.lstm.flatten_parameters()
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
            nn.LogSoftmax(dim=2),
        )

        
    def forward(self, inputs):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inputs)

        output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        
        out = self.mlp(output)

        return out

class Postagger_adaptive(nn.Module):
    def __init__(self, seq_length, emb_dim, hidden_size, output_size):
        super(Postagger_adaptive, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.lstm.flatten_parameters()
        self.out = nn.AdaptiveLogSoftmaxWithLoss(hidden_size, output_size, cutoffs=[round(output_size/5),2*round(output_size/5)], div_value=4)
        
    def forward(self, inputs, targets):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inputs)

        output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        output = output.contiguous().view(output.shape[0]*output.shape[1], output.shape[2])
        targets = targets.view(targets.shape[0]*targets.shape[1])

        return self.out(output, targets)

    def validation(self, inputs, targets):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inputs)

        output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        output = output.contiguous().view(output.shape[0] * output.shape[1], -1)
        targets = targets.view(targets.shape[0]*targets.shape[1])

        prediction = self.out.predict(output)
        _, loss = self.out(output, targets)

        return prediction, float(loss.cpu())