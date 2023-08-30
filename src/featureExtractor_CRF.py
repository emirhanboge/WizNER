from gazetteer_creator import create_gazetteer

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = stopwords.words('english')

gazetteer = create_gazetteer("../wikipedia_pages")

english_prefixes = {
    "anti": "",
    "auto": "",
    "down": "",
    "mega": "",
    "over": "",
    "post": "",
    "semi": "",
    "tele": "",
}

def shape_of(text):
  """
  Shape of a word
  AbcdE -> XxxxX
  """
  t1 = re.sub('[A-Z]', 'X',text) # replace all capital letters with X
  t2 = re.sub('[a-z]', 'x', t1) # replace all lowercase letters with x
  return re.sub('[0-9]', 'd', t2) # replace numbers with d

def short_shape_of(text):
  """
  Short shape of a word
  Xxx -> Xx
  """
  t1 = re.sub('[A-Z]', 'X',text) # replace all capital letters with X
  t2 = re.sub('[a-z]', 'x', t1) # replace all lowercase letters with x
  t3 = re.sub('[0-9]', 'd', t2) # replace numbers with d
  return re.sub('(.)\1+', '\g<1>', t3) # replace consecutive similar type letters with the first letter

def token2features(sentence: list, idx: int) -> dict:
  token = sentence[idx]

  # stem the word
  stem = PorterStemmer().stem(token[0])

  # POS tag
  pos = token[1]

  #Chunk tag
  chunk = token[2]

  #Start of the Sentence
  if idx == 0:
    start_of_sentence = True
  else:
    start_of_sentence = False

  # End of the Sentence
  if idx == len(sentence) - 1:
    end_of_sentence = True
  else:
    end_of_sentence = False

  # Starts with an uppercase letter
  if token[0][0].isupper():
    starts_with_uppercase = True
  else:
    starts_with_uppercase = False

  #𝑤𝑖’s shape
  shape = shape_of(token[0])

  #𝑤𝑖’s short word shape
  short_word_shape = short_shape_of(token[0])

  #𝑤𝑖 contains a number
  if any(char.isdigit() for char in token[0]):
    contains_number = True
  else:
    contains_number = False

  # wi contains a hyphen
  if '-' in token[0]:
    contains_hyphen = True
  else:
    contains_hyphen = False

  # wi is upper case and has a digit and a dash
  if starts_with_uppercase and contains_number and contains_hyphen:
    upper_case_with_digit_and_dash = True
  else:
    upper_case_with_digit_and_dash = False

  # wi contains a particular prefix (from all prefixes of length 4)
  global english_prefixes
  if any(token[0].startswith(prefix) for prefix in english_prefixes):
    contains_prefix = True
  else:
    contains_prefix = False

  # wi contains a particular suffix (from all suffixes of length 4)
  if len(token[0]) - len(stem) == 4:
    contains_suffix = True
  else:
    contains_suffix = False

  # wi is all uppercase
  if token[0].isupper():
    is_all_uppercase = True
  else:
    is_all_uppercase = False

  # Whether a stopword (you can use the nltk library)
  global stopwords
  if token[0] in stopwords:
    is_stopword = True
  else:
    is_stopword = False

  # Neighboring words
  if idx > 0:
    left_neighbor = sentence[idx-1][0]
  else:
    left_neighbor = ''
  if idx < len(sentence) - 1:
    right_neighbor = sentence[idx+1][0]
  else:
    right_neighbor = ''

  # Short word shape of neighboring words
  if idx > 0:
    left_neighbor_short_word_shape = short_shape_of(left_neighbor)
  else:
    left_neighbor_short_word_shape = ''
  if idx < len(sentence) - 1:
    right_neighbor_short_word_shape = short_shape_of(right_neighbor)
  else:
    right_neighbor_short_word_shape = ''

  # Word shape of neighboring words
  if idx > 0:
    left_neighbor_word_shape = shape_of(left_neighbor)
  else:
    left_neighbor_word_shape = ''
  if idx < len(sentence) - 1:
    right_neighbor_word_shape = shape_of(right_neighbor)
  else:
    right_neighbor_word_shape = ''

  global gazetteer
  if token[0] in gazetteer:
    is_in_gazetteer = True
  else:
    is_in_gazetteer = False

  # return a dictionary of features
  return {
    'stem': stem,
    'pos': pos,
    'chunk': chunk,
    'start_of_sentence': start_of_sentence,
    'end_of_sentence': end_of_sentence,
    'starts_with_uppercase': starts_with_uppercase,
    'shape': shape,
    'short_word_shape': short_word_shape,
    'contains_number': contains_number,
    'contains_hyphen': contains_hyphen,
    'upper_case_with_digit_and_dash': upper_case_with_digit_and_dash,
    'contains_prefix': contains_prefix,
    'contains_suffix': contains_suffix,
    'is_all_uppercase': is_all_uppercase,
    'is_stopword': is_stopword,
    'left_neighbor': left_neighbor,
    'right_neighbor': right_neighbor,
    'left_neighbor_short_word_shape': left_neighbor_short_word_shape,
    'right_neighbor_short_word_shape': right_neighbor_short_word_shape,
    'left_neighbor_word_shape': left_neighbor_word_shape,
    'right_neighbor_word_shape': right_neighbor_word_shape,
    'is_in_gazetteer': is_in_gazetteer
  }

def sent2features(sentence):
    return [token2features(sentence, i) for i in range(len(sentence))]

def sent2labels(sentence):
    return [label for x, y, z, label in sentence]

