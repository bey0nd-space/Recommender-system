from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
text=['my name is rajat rajat','hello friends chai peelo']
tokenized_doc=[word_tokenize(doc.lower()) for doc in text]
dictionary=Dictionary(tokenized_doc)
dictionary.token2id
corpus=[dictionary.doc2bow(doc) for doc in tokenized_doc]
corpus
tfidf=TfidfModel(corpus)
tfidf[corpus[0]]


num_features=300
min_word_count=1
num_workers=4
context=10
downsampling=1e-3
from gensim.models import Word2Vec
print("Training model....")
model = word2vec.Word2Vec(sen_corpus, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
corpus=[
    'all cats are my love','meera is affection','but cats are affection']
vectorizer=CountVectorizer()
features=vectorizer.fit_transform(corpus).todense()
print(vectorizer.vocabulary_)
for f in features:
    print(euclidean_distances(features[0],f))


import pandas as pd
df=pd.read_json("C:/Users/Inspi/Desktop/mlproject/DEMO/pickels/data.json")
df.shape

training_set=pd.DataFrame(df).set_index('asin')[:128197].copy(deep=True)
training_set

test_set=pd.DataFrame(df).set_index('asin')[128197:].copy(deep=True)
test_set

from nltk.tokenize import word_tokenize 
text=test_set['title'].apply(word_tokenize)
text
