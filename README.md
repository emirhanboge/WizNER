# WizNER: Named Entity Recognition System for English News Articles 📰🔍

This project is focused on Named Entity Recognition (NER) for English news articles and consists of two major components:

## Table of Contents
1. [Gazetteer Implementation](#gazetteer-implementation)
2. [Machine Learning Models](#machine-learning-models)
3. [Dataset](#dataset)
4. [Entity Types](#entity-types)
5. [File Structure](#file-structure)
6. [Report](#report)

## Gazetteer Implementation 📖
The first component involves the creation of a gazetteer, implemented using regular expressions and rules derived from Wikipedia pages for identifying various types of named entities.

- [`gazetteer_creator.py`](src/gazetteer_creator.py): Python script responsible for generating the gazetteer using Wikipedia data.

## Machine Learning Models 🤖
The second component of the project explores two machine learning models for NER:

- Conditional Random Fields (CRF)
- Recurrent Neural Networks (RNN)

### Conditional Random Fields
Implemented in [`CRF.py`](src/CRF.py).

### Recurrent Neural Networks
Implemented in [`rnn.py`](src/RNN.py).

## Dataset 📊
The dataset consists of news articles from the Reuters Corpus. [`dataset`](dataset)

## Entity Types 🏷
The system is capable of identifying the following types of entities:

- Person 👤
- Location 🌍
- Organization 🏢
- Miscellaneous: Entities not classified as Person, Location, or Organization.

## File Structure 📂

- [`CRF.py`](src/CRF.py): Houses the Conditional Random Fields model.
- [`data_reader.py`](src/data_reader.py): Responsible for reading the dataset and preparing it for model training and evaluation.
- [`featureExtractor_CRF.py`](src/featureExtractor_CRF.py): Handles feature extraction for the CRF model.
- [`rnn.py`](src/RNN.py): Contains the Recurrent Neural Network model implementation.
- [`data_processing.py`](src/data_processing.py): Includes various data pre-processing utilities.
- [`embeddings.py`](src/embeddings.py): Manages the training and usage of word embeddings.

## Report 📄
For a comprehensive understanding of the methodologies, results, and other relevant details, refer to [`report.md`](report.md).










