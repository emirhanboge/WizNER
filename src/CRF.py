from data_reader import read_data
from featureExtractor_CRF import sent2features, sent2labels

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import sklearn_crfsuite
import pandas as pd

def get_features_for_sents(sents, features):
    feature_data = []
    for sent in sents:
        sent_data = []
        for token in sent:
            d = {f: token[f] for f in features}
            sent_data.append(d)
        feature_data.append(sent_data)
    return feature_data

def perform_grid_search(train_data, train_labels):
    grid = GridSearchCV(estimator=sklearn_crfsuite.CRF(),
                        param_grid={'c1': [1, 10],
                                    'c2': [1, 10],
                                    'max_iterations': [100, 300],
                                    'algorithm': ['lbfgs', 'l2sgd', 'ap', 'arow']},
                        cv=5)
    grid.fit(train_data, train_labels)
    return grid

def calculate_metrics(y_true, y_pred):
    precision = metrics.flat_precision_score(y_true, y_pred, average='macro')
    recall = metrics.flat_recall_score(y_true, y_pred, average='macro')
    f1 = metrics.flat_f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

if __name__ == '__main__':
    PATH = "../dataset"

    train_data = read_data(f'{PATH}/train.txt')
    val_data = read_data(f'{PATH}/valid.txt')
    test_data = read_data(f'{PATH}/test.txt')

    train_sents = [sent2features(s) for s in train_data]
    val_sents = [sent2features(s) for s in val_data]
    test_sents = [sent2features(s) for s in test_data]

    train_labels = [sent2labels(s) for s in train_data]
    val_labels = [sent2labels(s) for s in val_data]
    test_labels = [sent2labels(s) for s in test_data]

    features_list = ['stem', 'pos', 'chunk', 'start_of_sentence', 'end_of_sentence',
                     'starts_with_uppercase', 'shape', 'short_word_shape',
                     'contains_number', 'contains_hyphen', 'upper_case_with_digit_and_dash',
                     'contains_prefix', 'contains_suffix', 'is_all_uppercase', 'is_stopword',
                     'left_neighbor', 'right_neighbor', 'left_neighbor_short_word_shape',
                     'right_neighbor_short_word_shape', 'left_neighbor_word_shape',
                     'right_neighbor_word_shape', 'is_in_gazetteer']

    increasing_features = [features_list[:i] for i in range(1, len(features_list) + 1)]

    results = pd.DataFrame(columns=['Features', 'Precision', 'Recall', 'F1 Score'])

    for i, features in enumerate(increasing_features):
        train_feature_data = get_features_for_sents(train_sents, features)
        val_feature_data = get_features_for_sents(val_sents, features)

        grid = perform_grid_search(train_feature_data, train_labels)

        y_pred = grid.predict(val_feature_data)
        precision, recall, f1 = calculate_metrics(val_labels, y_pred)

        last_feature = '+' + features[-1] if i != 0 else features[-1]
        results = results.append({'Features': last_feature,
                                  'Precision': precision,
                                  'Recall': recall,
                                  'F1 Score': f1}, ignore_index=True)

    best_features = ['stem']
    for i in range(1, len(results)):
        if results.iloc[i]['F1 Score'] > results.iloc[i-1]['F1 Score']:
            best_features.append(results.iloc[i]['Features'].lstrip('+'))

    train_best_features = get_features_for_sents(train_sents, best_features)
    test_best_features = get_features_for_sents(test_sents, best_features)

    crf = sklearn_crfsuite.CRF(c1=grid.best_params_['c1'],
                               c2=grid.best_params_['c2'],
                               max_iterations=grid.best_params_['max_iterations'],
                               algorithm=grid.best_params_['algorithm'])

    crf.fit(train_best_features, train_labels)
    y_pred = crf.predict(test_best_features)

    precision, recall, f1 = calculate_metrics(test_labels, y_pred)

    print(f'Best features: {best_features}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


