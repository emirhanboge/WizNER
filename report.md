# Report:

## Read Dataset:
Train, Validation, Test sets are imported and preprocessed.
Preprocessing includes extracting the words, pos tags, chunk tags, labels (NER) from the txt files.
The extracted words, pos tags, chunk tags, labels are stored in lists of sentences.

## Create Gazetter:
The gazetter is created from the given wikipedia dataset, which is 20000 pages long, and the words that are in the wikipedia pages' internal links are added to the gazetter.
Getting the internal links is done by using the following method:
1. Get the page's text
2. Find the all href links in the text by using the following regular expression pattern:
  - href="([^"]+)"
3. Clean the links by applying:
    1. Replacing special characters with their associated unicode characters
    2. Removing some text such as
    3. Finding the anchor tag by using regular expression:

Only the unique words are kept in the gazetter, words that start with a lower case letter are removed, words that are 1 character long are removed, and non-alphanumeric characters are removed.

After gazetteer is created,

Size of the gazetter: 365214

## Conditional Random Field (CRF):

Following features are extracted from the given dataset to create the features for the CRF:
- Stem of the word
- Part of speech tag
- Chunk tag
- Start of the sentence, type: boolean
- End of the sentence, type: boolean
- Starts with an uppercase letter, type: boolean
- Wi's shape
- Wi's short word shape
- Wi contains a number type: boolean
- wi contains a hyphen type: boolean
- wi is upper case and has a digit and a dash, type: boolean
- wi contains a particular prefix (from all prefixes of length 4), type: boolean
- wi contains a particular suffix (from all suffixes of length 4), type: boolean
- wi is all uppercase, type: boolean
- Whether a stopword (you can use the nltk library), type: boolean
- Left word
- Right word
- Left word shape
- Right word shape
- Short word shape of left word
- Short word shape of right word
- Presence of wi in the gazetteer, type: boolean


What is meant by word shape is that making the characters of the word X or x or d, and short word shape is the same as word shape, but without recurring Xs xs or ds.


After the features are applied to the tokens of the sentences, the labels are also extracted from the sentences.
Then the CRF model is trained on the features and labels as the following:
1. Cross-validation is used to find the best parameters for the model with the help of GridSearchCV.
  Parameters:
    - max_iter: maximum number of iterations [100, 300]
    - c1: L1 regularization strength [1, 10]
    - c2: L2 regularization strength [1, 10]
    - algorithm: algorithm to use ['lbfgs', 'l2sgd', 'ap', 'arow']
2. Features are fed into the model one by one and the results of each feature are stored in a dataframe called results.
  Results on the validation set, + means that it has all the features upto that point:
      - Features	Precision	Recall	F1 Score
      - 0	stem	0.775165	0.525914	0.608795
      - 1	+pos	0.774851	0.714408	0.735613
      - 2	+chunk	0.781901	0.711504	0.738660
      - 3	+start_of_sentence	0.783442	0.721279	0.745544
      - 4	+end_of_sentence	0.784116	0.717293	0.742632
      - 5	+starts_with_uppercase	0.789376	0.760176	0.770464
      - 6	+shape	0.790433	0.761493	0.773555
      - 7	+short_word_shape	0.790368	0.765983	0.776180
      - 8	+contains_number	0.791281	0.763569	0.775122
      - 9	+contains_hyphen	0.795059	0.763601	0.776799
      - 10 +upper_case_with_digit_and_dash	0.793233	0.764014	0.776290
      - 11 +contains_prefix	0.795360	0.764592	0.777586
      - 12 +contains_suffix	0.791346	0.762783	0.774732
      - 13 +is_all_uppercase	0.793458	0.765239	0.777089
      - 14 +is_stopword	0.793150	0.763003	0.775697
      - 15 +left_neighbor	0.826686	0.797247	0.810402
      - 16 +right_neighbor	0.846928	0.812609	0.828235
      - 17 +left_neighbor_short_word_shape	0.853478	0.815648	0.832896
      - 18 +right_neighbor_short_word_shape	0.862449	0.825643	0.842405
      - 19 +left_neighbor_word_shape	0.862513	0.826031	0.842529
3. The model is trained on the training set using the best parameters and best features found in the cross-validation step.

    Best parameters:
        - max_iter: 300
        - c1: 1
        - c2: 1
        - algorithm: lbfgs
    Best features are extracted from the results dataframe by taking into account if that feature has improved the F1 score or not.

4. The model is tested on the test set and the results are as follows:

  Classification Report and F1 Score:
                precision    recall  f1-score   support

        B-LOC       0.81      0.84      0.82      1668
        B-MISC      0.80      0.70      0.74       702
        B-ORG       0.80      0.71      0.75      1661
        B-PER       0.79      0.82      0.81      1617
        I-LOC       0.81      0.65      0.72       257
        I-MISC      0.54      0.64      0.59       216
        I-ORG       0.67      0.73      0.70       835
        I-PER       0.83      0.94      0.88      1156
            O       0.99      0.99      0.99     38555

        accuracy                        0.95     46667
        macro avg   0.78      0.78      0.78     46667
        weighted avg 0.95     0.95      0.95     46667

        F1 Score: 0.7784959923215771


## Reccurent Neural Network (RNN):

Steps Explained:
1. The train, test, and validation sets are divided into sentences, and labels
2. The labels that are in the dataset are converted to their associated unique integers:
    - 'Other': 0,
    - 'B-LOC': 1,
    - 'B-MISC': 2,
    - 'B-ORG': 3,
    - 'B-PER': 4,
    - 'I-LOC': 5,
    - 'I-MISC': 6,
    - 'I-ORG': 7,
    - 'I-PER': 8
    - 'O': 9

    Other is used for the labels that are not in the dataset and used in the padding step as dummy labels.

3. First Word embedding is created by the words that are in the dataset and the second word emmbedding has the words imported from the gensim library.
  - Words are converted to lowercase to avoid errors in the embedding model.
4. The sentences are converted to lists of tokens as follows:
    - Tokenizer is created to split the sentences into tokens.
    - Sentences of the train set are fit to the tokenizer.
    - Sentences of train, test, and validation sets are converted to sequences to be used in the model.
    - All sequences are padded to the same length to maintain consistency in the model.
    - Labels are also converted to sequences and padded to the same length as the other sequences.
        Padding style is 'post' which is used to add padding to the end of the sequence, this is done to get the non-padded part of the sequence in the end to evaluate the model.
5. Embeddings Layers are created to embed the tokens into vectors using the word embeddings.

    Used Following Embedding Strategies:
      * Randomly Initialized Embedding Layer
      * Self-trained Embedding Layer created by our corpus
          * Trainable to allow the embedding layer to be trained while the model is being trained.
          * Non-trainable to prevent the embedding layer from being trained while the model is being trained.
          * Masking used to prevent the embedding layer from being trained on the padding tokens in the sequences to prevent the embedding layer from being trained on the padding tokens.
      * Pre-trained Embedding Layer from the gensim library
          * Trainable to allow the embedding layer to be trained while the model is being trained.
          * Non-trainable to prevent the embedding layer from being trained while the model is being trained.
          * Masking used to prevent the embedding layer from being trained on the padding tokens in the sequences to prevent the embedding layer from being trained on the padding tokens.
          * If the word is not in the embedding layer, that index in the embedding layer is initialized with random values.

6. Model is created with the following methods:
    - Hyperparameters
      - Embedding Layers:
          - Randomly Initialized Embedding Layer
          - Self-trained Embedding Layer created by our corpus which is trainable
          - Self-trained Embedding Layer created by our corpus which is non-trainable
          - Pre-trained Embedding Layer from the gensim library which is trainable
          - Pre-trained Embedding Layer from the gensim library which is non-trainable
      - Hidden Layer Sizes: 100, 50
    - Layers of the model:
        - Embedding Layer, which is the layer where word embeddings are added
        - Bidirectional LSTM Layer, which is the layer where the sentences are processed by doing forward and backward passes of the BidirectionalLSTM, Bidirectional allows the model to process the sentences in both directions.
        - Dense Layer, which is the output layer with the softmax activation function. This is the layer where the model will predict the labels.
    - Sparse Categorical Crossentropy is used as the loss function for the model to train the model because the labels are sparse categorical.
    - Adam is used as the optimizer for the model to train the model.
7. The best model which has the highest validation accuracy is extracted to be used in the final model.
    - Best Embedding Layer: Trainable pre-trained embedding layer from the gensim library
    - Best hidden layer size: 50
8. The best model predicts the labels of the test set.
9. The labels are converted to their original form by using the align_predictions function.
    - Non-padded part of the sequence is extracted from the test and prediction sequences to get the original labels and to evaluate the model.
    - The labels are converted to their original form.
10. Model is evaluated by the seqeval library, metrics include precision, recall, f1-score, and classification report. Results are as follows:

    Distribution of the predicted labels:
    'O': 41448, 'B-LOC': 1592, 'B-MISC': 542, 'I-LOC': 138, 'I-MISC': 153, 'B-PER': 884, 'I-PER': 360, 'B-ORG': 1158, 'I-ORG': 392

    True Labels:
    'O': 38555, 'B-LOC': 1668, 'B-PER': 1617, 'I-PER': 1156, 'I-LOC': 257, 'B-MISC': 702, 'I-MISC': 216, 'B-ORG': 1661, 'I-ORG': 835

    The distribution of the predicted labels is proportional to the true labels.

    LOC, MISC, ORG, PER evaluation, O not included in the evaluation:
              precision    recall  f1-score   support

        LOC     0.4058    0.3939    0.3998      1668
        MISC    0.4084    0.3462    0.3747       702
        ORG     0.4444    0.3516    0.3926      1661
        PER     0.2749    0.1602    0.2024      1617

        micro avg     0.3899    0.3086    0.3445      5648
        macro avg     0.3834    0.3130    0.3424      5648
        weighted avg  0.3800    0.3086    0.3380      5648

        F1: 0.34237383791217846
        Precision: 0.38340019508798207
        Recall: 0.312951830714111

  Model gives low F1, precision, recall scores because the LOC, MISC, ORG, and PER labels are not much in the given training set so the model is not able to predict them well.

