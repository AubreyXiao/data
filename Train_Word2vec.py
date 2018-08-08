import logging
from gensim.models import word2vec
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import nltk
import re
from nltk import tokenize
from MasterProject.data_Preprocessing.Datasets import get_data
import pickle
#get the train_data,
data_manipulation = get_data.Datasets()

def clean_text_to_text(review):
    # remove the the',",\
    review = re.sub(r"\\", "", review)
    review = re.sub(r"\'", "", review)
    review = re.sub(r"\"", "", review)
    # return the lower case
    text = review.strip().lower()

    return text


def get_normalized_data(data):
    reviews = []
    review_sentences = []
    review_tokens = []

    # clean the test dataset
    for review in data["Column2"]:
        text = BeautifulSoup(review)
        cleaned_text = text.get_text().encode('ascii', 'ignore')
        cleaned_string = cleaned_text.decode('utf-8')
        cleaned_review = clean_text_to_text(cleaned_string)
        reviews.append(cleaned_review)
        sentence = tokenize.sent_tokenize(cleaned_review)
        # number of the review
        review_sentences.append(sentence)

        for s in sentence:
            if (len(s) > 0):
                tokens = text_to_word_sequence(s)
                # filter out non-alpha
                tokens = [token for token in tokens if token.isalpha()]
                # filter out those short letters
                tokens = [t for t in tokens if len(t) > 1]
                review_tokens.append(tokens)

    return reviews, review_sentences, review_tokens

train_reviews, train_sentences, train_tokens = data_manipulation.get_normalized_data("train")
unsup_reviews, unsup_sentences, unsup_tokens = data_manipulation.get_normalized_data("unlabeled")
test_reviews, test_sentences, test_tokens= data_manipulation.get_normalized_data("test")
po_reviews, po_sentences , po_tokens = data_manipulation.get_normalized_data("polarity")


#------add extra dataset-------
extra_train = pd.read_csv("/Users/xiaoyiwen/Desktop/MasterProject/MasterProject/data_Preprocessing/Datasets/kaggle_data/training_set.csv")


extra_test = pd.read_csv("/Users/xiaoyiwen/Desktop/MasterProject/MasterProject/data_Preprocessing/Datasets/kaggle_data/test_set.csv")


reviews1, sentences1, tokens1 = get_normalized_data(extra_train)

reviews2, sentences2, tokens2 = get_normalized_data(extra_test)




#--------add poalarity set----------------------

#-------print all the size of data ---------------

all_sentences = train_tokens + unsup_tokens + test_tokens + po_tokens +tokens1 + tokens2

#
# print(reviews1)
# print(tokens1)
#
# print(reviews2)
# print(tokens2)
#
# print(len(reviews1))
# print(len(sentences1))
#
# print(len(reviews2))
# print(len(sentences2))

print(len(all_sentences))
#----buildup word2vec---------

#300features_10min_5window_NEG10_polarity_extra
#set the format
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#create a model
num_features = 300 #dimensionality
min_word_count = 10 #filter frequence>40
num_workers = 8 # threads to run in parrell
context = 10 #contect window size
downsampling = 1e-3
print("Training models!")

model = word2vec.Word2Vec(all_sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling,negative=10)

#init_sims wiil make the model memory efficient
model.init_sims(replace=True)

#save the modle
filename = "300features_10min_15window_NEG10_polarity_extra"
model.save(filename)

#save the modle in text form
filename1 = '300features_10min_15window_NEG10_polarity_extra.txt'
model.wv.save_word2vec_format(filename1,binary=False)
