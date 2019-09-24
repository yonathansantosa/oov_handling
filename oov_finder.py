from nltk.corpus import brown
import torch
import os

from model import *
from wordembedding import Word_embedding

from tagset.tagset import Postag, Postagger, Postagger_adaptive
import argparse

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--lang', default='en', help='choose which language for word embedding')

args = parser.parse_args()

word_embedding = Word_embedding(lang=args.lang, embedding=args.embedding, freeze=True, sparse=False)
dataset = Postag(word_embedding)
tagged_words = set([word for word, _ in dataset.tagged_words])

f = open(f'oov_list.txt_{args.embedding}', 'w')

for word in tagged_words:
    if word not in word_embedding.stoi:
        f.write(f'{word} ')