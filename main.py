from gensim.models import Word2Vec
import gensim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

nltk.download('punkt_tab')

# Function to load and preprocess the dataset 

def loadAndPreProcess(path):
  
  df = pd.read_json(path, lines = True) # Read the data

  df["label"] = df.annotation.apply(lambda x: x.get('label'))  # Extract label list
  df["label"] = df.label.apply(lambda x: x[0])  # Get first label from the list
    
  X = df.content.values  # Extract text content
  y = df.label.values  # Extract labels
  
  return X, y


# Load the data

X, y = loadAndPreProcess('/content/Dataset.json')

# Store them in a new list

data = []

# Browse (Interate the data) and add store them into in a list
for text in pd.Series(X).dropna(): # Using dropana to avoid NaN Errors
  sentences = sent_tokenize(text)
  # Tokenize the sentences into a words and apply lower function
  for sent in sentences:
    words = [word.lower() for word in word_tokenize(sent)]
    data.append(words)

# Using Skip-Gram as Word Embbeding technics 

SkipGramModel = Word2Vec(
    sentences= data, 
    vector_size= 150, 
    window= 7, 
    sg= 1, 
    min_count= 1, 
    workers= 4,
    )

# Training
SkipGramModel.train(data, total_examples= len(data), epochs= 20)

# Saving
#SkipGramModel.save("SkipGramModel.bin")

print(SkipGramModel.wv.most_similar("fuck", topn=5))

print()

print(SkipGramModel.wv["fuck"])
