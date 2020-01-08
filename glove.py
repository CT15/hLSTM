import bcolz
import pickle
import torch.nn as nn
import numpy as np
from collections import Counter

class Glove():
    def __init__(self, unk_token='<unk>', pad_token='<pad>'):
        self.emb_dim = 300

        vectors = bcolz.open(f'../glove.6B/extracted/glove.6B.300d.dat')[:]
        self.words = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_words.pkl', 'rb'))
        
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.words = [unk_token, pad_token] + self.words
 
        self.word2idx = {o:i for i, o in enumerate(self.words)}
        self.idx2words = {i:o for i, o in enumerate(self.words)}

        self.embedding = {w: vectors[word2idx[w]] for w in words}
        self.embedding[unk_token] = np.random.normal(scale=0.6, size=(self.emb_dim, ))
        self.embedding[pad_token] = np.zeros((self.emb_dim,))

    # TODO: I think this method is very shitty
    def create_custom_embedding(self, sentences):
        # remove words that appear only once (likely typo)
        words = Counter()
        for sentence in enumerate(sentences):
            for word in sentence.split(" "):
                words.update([word.lower()]) # lower case
        self.words = {k:v for k,v in words.items() if v > 1}
        
        # sort words => most common words first
        self.words = sorted(words, key=words.get, reverse=True)

    # TODO: I think this method is very shitty
    def add_to_embedding(self, tokens, vector=None):
        if vector is not None and vector.shape != (self.emb_dim, ):
            raise Exception(f'Expected vector size of ({self.emb_dim}, ), get {vector.shape} instead.')

        for token in tokens:
            embedding[token] = np.random.normal(scale=0.6, size=(self.emb_dim, )) if vector is None else vector
            if token not in self.words:
                self.words.append(token)

        self.word2idx = {o:i for i, o in enumerate(self.words)}
        self.idx2words = {i:o for i, o in enumerate(self.words)}


    @property
    def weights_matrix(self, unk_token):
        if unk_token not in self.words:
            raise Exception(f'{unk_token} is not registered. Add token to embedding first.')

        weights_matrix = np.zeros((len(self.idx2words), self.emb_dim))

        for i, word in idx2words.items():
            try: 
                weights_matrix[i] = self.embedding[word]
            except KeyError: # found an unknown word
                weights_matrix[i] = self.embedding[unk_token]
                self.embedding[word] = weights_matrix[i]

        return weights_matrix


    def sentence_to_indices(self, sentence, pad=True, seq_len=-1):
        # if pad is False, seq_length is ignored
        if pad and seq_len <= 0:
            raise Exception('Invalid seq_len. Must be positive integer.')

        word_list = sentence.split(" ")
        word_list = [self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_token] for word in word_list]

        if pad and seq_len - len(word_list) > 0:
            word_list.append(self.word2idx[self.pad_token] * (seq_len - len(word_list)))

        return word_list


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    emb_layer.weight = nn.Parameter(weights_matrix)

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer
 