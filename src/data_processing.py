from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_sentences_and_labels(data):
    sentences = []
    labels = []
    for sent in data:
        sentence = []
        label = []
        for token in sent:
            sentence.append(token[0])
            label.append(token[3])
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def labels_to_ids(labels, label_to_id):
    labels_id = []
    for sentence_labels in labels:
        labels_id.append([label_to_id[label] for label in sentence_labels])
    return labels_id

def get_max_length(sequences):
    return max(len(seq) for seq in sequences)

def prepare_sequences(tokenizer, sequences, max_length):
    sequences = tokenizer.texts_to_sequences(sequences)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

def prepare_labels(labels, max_length):
    return pad_sequences(labels, maxlen=max_length, padding='post')

