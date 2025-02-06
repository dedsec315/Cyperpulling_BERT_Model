from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pandas as pd

warnings.filterwarnings(action='ignore')

# Read the data

df = pd.read_json('./Dataset/dataset.json')


# See the data

df.head()

# Tokenize the data

data = df['text'].apply(word_tokenize)

