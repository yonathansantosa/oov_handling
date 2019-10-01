# coding: utf-8
from __future__ import division

import struct
import sys
import gzip
import binascii
import re

import argparse

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--max', default=3000000)
parser.add_argument('--embdim', default=300)
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--negative', default=False, action='store_true')
parser.add_argument('--split', default=False, action='store_true')
args = parser.parse_args()

if args.local:
    FILE_NAME = "./embeddings/word_embedding/GoogleNews-vectors-negative300.bin.gz" # outputs GoogleNews-vectors-negative300.bin.gz.txt
    SAVE_TO = "./embeddings/word_embedding/GoogleNews-vectors-negative300.bin.gz"
    CHUNK_SIZE = "./embeddings/word_embedding/chunk-size.txt"
    STOI = "./embeddings/word_embedding/stoi.txt"
else:
    FILE_NAME = "/content/gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz" # outputs GoogleNews-vectors-negative300.bin.gz.txt
    SAVE_TO = "./embeddings/word_embedding/GoogleNews-vectors-negative300.bin.gz"
    CHUNK_SIZE = "./embeddings/word_embedding/chunk-size.txt"
    STOI = "./embeddings/word_embedding/stoi.txt"

MAX_VECTORS = int(args.max) # Top words to take
FLOAT_SIZE = 4 # 32bit float
emb_dim = int(args.embdim)
split = 100000
output_file_name = SAVE_TO
file_id = 0
file_type = '.txt'
counter = 0
idxs = 0
if args.split:
    f_out = open('%s-%d%s' % (output_file_name, file_id, file_type), 'w', encoding='utf-8')
else:
    f_out = open('%s%s' % (output_file_name, file_type), 'w', encoding='utf-8')
    
with gzip.open(FILE_NAME, 'rb') as f, open(STOI, 'w', encoding='utf-8') as stoi, open(CHUNK_SIZE, 'w') as chunk_s:
    c = None
    # read the header
    header = ""
    while c != b"\n":
        c = f.read(1)
        header += c.decode('utf8')

    total_num_vectors, vector_len = (int(x) for x in header.split())
    num_vectors = min(MAX_VECTORS, total_num_vectors)

    print("Taking embeddings of top %d words (out of %d total)" % (num_vectors, total_num_vectors))
    print("Embedding size: %d" % emb_dim)

    for j in range(num_vectors):
        if counter >= split and args.split:
            f_out.close()
            chunk_s.write('%d\n' % counter)
            counter = 0
            file_id += 1
            f_out = open('%s-%d%s' % (output_file_name, file_id, file_type), 'w', encoding='utf-8')
        word = ""
        cont = False
        while True:
            # c = binascii.hexlify(c)
            c = f.read(1) if not cont else c + f.read(1)
            if c == b" ":
                break
            try:
                c.decode('utf8')
                cont = False
            except UnicodeDecodeError:
                cont = True
                continue
            word += c.decode('utf8', 'ignore')
        
        binary_vector = f.read(FLOAT_SIZE * vector_len)
        txt_vector = [ "%s" % struct.unpack_from('f', binary_vector, i)[0] 
                   for i in range(0, len(binary_vector), FLOAT_SIZE) ]
        txt_vector = txt_vector[:emb_dim]
        # print(txt_vector)
        
        if args.negative:
            f_out.write("%s %s\n" % (word, " ".join(txt_vector)))
            stoi.write('%s %d %d\n' % (word, file_id, counter))
            counter += 1
        elif re.findall('_', word) == [] and re.findall('http', word) == [] and re.findall('.com', word) == []:
            f_out.write("%s %s\n" % (word, " ".join(txt_vector)))
            stoi.write('%s %d %d\n' % (word, file_id, counter))
            counter += 1
        
        sys.stdout.write("%d%%\r" % ((j + 1) / num_vectors * 100))
        sys.stdout.flush()
        
        if (j + 1) == num_vectors:
            break
        
f_out.close()
chunk_s = open(CHUNK_SIZE, 'a')
chunk_s.write('%d\n' % counter)
chunk_s.close()
print("\nDONE!")
print("Output written to %s" % output_file_name)