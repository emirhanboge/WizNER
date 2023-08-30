from data_processing import get_sentences_and_labels, labels_to_ids, get_max_length, prepare_sequences, prepare_labels
from embeddings import train_own_embeddings, train_pretrained_embeddings, create_embedding_layer

def prepare_data(train_data, val_data, test_data, label_to_id, tokenizer, max_length):
    train_X, train_y = get_sentences_and_labels(train_data)
    val_X, val_y = get_sentences_and_labels(val_data)
    test_X, test_y = get_sentences_and_labels(test_data)

    train_y_id = labels_to_ids(train_y, label_to_id)
    val_y_id = labels_to_ids(val_y, label_to_id)
    test_y_id = labels_to_ids(test_y, label_to_id)

    train_X = prepare_sequences(tokenizer, train_X, max_length)
    val_X = prepare_sequences(tokenizer, val_X, max_length)
    test_X = prepare_sequences(tokenizer, test_X, max_length)

    train_y_id = prepare_labels(train_y_id, max_length)
    val_y_id = prepare_labels(val_y_id, max_length)
    test_y_id = prepare_labels(test_y_id, max_length)

    return train_X, train_y_id, val_X, val_y_id, test_X, test_y_id, test_y

def create_embedding_layers(own_embed, pre_embed, tokenizer, max_words, embedding_dim, max_length):
    own_embedding_layer = create_embedding_layer(own_embed, tokenizer, max_words, embedding_dim, max_length)
    own_embedding_layer_trainable = create_embedding_layer(own_embed, tokenizer, max_words, embedding_dim, max_length, trainable=True)
    pre_embedding_layer = create_embedding_layer(pre_embed, tokenizer, max_words, embedding_dim, max_length)
    pre_embedding_layer_trainable = create_embedding_layer(pre_embed, tokenizer, max_words, embedding_dim, max_length, trainable=True)

    random_embedding_matrix = np.random.rand(max_words, embedding_dim)
    random_embedding_layer = create_embedding_layer(random_embedding_matrix, tokenizer, max_words, embedding_dim, max_length)

    return own_embedding_layer, own_embedding_layer_trainable, pre_embedding_layer, pre_embedding_layer_trainable, random_embedding_layer

def align_predictions(predictions, label_ids, id_to_label, test_y):
  new_predictions = []
  for i in range(len(predictions)):
    length_of_sentence = len(test_y[i])
    new_predictions.append(predictions[i][:length_of_sentence])

  test_predictions = []
  test_true = []
  for i in range(len(new_predictions)):
    test_predictions.append([])
    test_true.append([])
    for j in range(len(new_predictions[i])):
      test_predictions[i].append(id_to_label[new_predictions[i][j]])
      test_true[i].append(id_to_label[label_ids[i][j]])

  return test_predictions, test_true

def create_and_train_models(train_X, train_y_id, val_X, val_y_id, embeds, hidden_layer_sizes, label_to_id):
    models = []
    tried_embed_layers = []
    tried_hidden_layers = []
    val_accs = []

    for hidden_layer_size in hidden_layer_sizes:
        index = 0

        for embedding_layer in embeds:
            callback = ModelCheckpoint('model_' + str(hidden_layer_size) + '_' + str(index) + '.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
            model = Sequential()
            model.add(embedding_layer)
            model.add(Bidirectional(LSTM(hidden_layer_size, return_sequences=True)))
            model.add(Dense(len(label_to_id), activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(train_X, train_y_id, epochs=2, batch_size=32, validation_data=(val_X, val_y_id), callbacks=[callback])
            model.load_weights('model_' + str(hidden_layer_size) + '_' + str(index) + '.h5')
            val_accuracy = model.evaluate(val_X, val_y_id, verbose=0)[1]
            val_accs.append(val_accuracy)
            models.append(model)
            tried_hidden_layers.append(hidden_layer_size)
            index += 1

    return models, tried_embed_layers, tried_hidden_layers, val_accs

if __name__ == '__main__':
    train_data = open('data/train.txt').read().strip().split('\n\n')
    val_data = open('data/valid.txt').read().strip().split('\n\n')
    test_data = open('data/test.txt').read().strip().split('\n\n')

    label_to_id = {'Other': 0, 'B-LOC': 1, 'B-MISC': 2, 'B-ORG': 3, 'B-PER': 4, 'I-LOC': 5, 'I-MISC': 6, 'I-ORG': 7, 'I-PER': 8, 'O': 9}
    id_to_label = {v: k for k, v in label_to_id.items()}

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_X)
    max_length = get_max_length(train_X + val_X + test_X)

    train_X, train_y_id, val_X, val_y_id, test_X, test_y_id, test_y  = prepare_data(train_data, val_data, test_data, label_to_id, tokenizer, max_length)

    train_X = [[word.lower() for word in sentence] for sentence in train_X]

    own_embed = train_own_embeddings(train_X)
    pre_embed = train_pretrained_embeddings('text8')

    embedding_dim = 100
    max_words = len(tokenizer.word_index) + 1
    own_embedding_layer, own_embedding_layer_trainable, pre_embedding_layer, pre_embedding_layer_trainable, random_embedding_layer = create_embedding_layers(own_embed, pre_embed, tokenizer, max_words, embedding_dim, max_length)

    embeds = [own_embedding_layer, own_embedding_layer_trainable, pre_embedding_layer, pre_embedding_layer_trainable, random_embedding_layer]
    hidden_layer_sizes = [32, 64, 128, 256]

    models, tried_embed_layers, tried_hidden_layers, val_accs = create_and_train_models(train_X, train_y_id, val_X, val_y_id, embeds, hidden_layer_sizes, label_to_id)

    best_model_index = val_accs.index(max(val_accs))
    best_model = models[best_model_index]
    best_embedding_layer = tried_embed_layers[best_model_index]
    best_hidden_layer_size = tried_hidden_layers[best_model_index]

    test_pred = best_model.predict(test_X)
    test_pred = np.argmax(test_pred, axis=2)

    test_predictions, test_true = align_predictions(test_pred, test_y_id, id_to_label, test_y)

    print(classification_report(test_true, test_predictions, digits=4))
    print('F1:', f1_score(test_true, test_predictions, average='macro'))
    print('Precision:', precision_score(test_true, test_predictions, average='macro'))
    print('Recall:', recall_score(test_true, test_predictions, average='macro'))


