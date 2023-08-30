def read_data(filename) -> list:
  data = open(filename, 'r').read().replace('\n\n', ' <eos> ')
  sentences = data.split('<eos>')
  data = []
  for sentence in sentences:
      tokens = sentence.split('\n')

      data2 = []
      for token in tokens:
          token = token.split():
          try:
            word, pos, chunk, ner = token[0], token[1], token[2], token[3]
          except:
            pass
          data2.append((word, pos, chunk, ner))
      data.append(data2)

  return data
