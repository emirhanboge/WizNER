import numpy as np
from tensorflow.keras.layers import Embedding
from gensim.models import Word2Vec
from gensim import downloader as api

def train_own_embeddings(sentences):
    sentences = [[word.lower() for word in sentence] for sentence in sentences]
    own_embed = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return own_embed

def train_pretrained_embeddings(corpus_name):
    corpus = api.load(corpus_name)
    return Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

def create_embedding_layer(embed_model, tokenizer, max_words, embedding_dim, max_length, trainable=True):
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_matrix[i] = embed_model.wv[word]
        except:
            embedding_matrix[i] = np.random.rand(embedding_dim)
    return Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=trainable, mask_zero=True)

