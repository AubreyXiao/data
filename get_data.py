import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import sys
import os
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import nltk
import pickle
import chardet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize

class Datasets():
   lemma = WordNetLemmatizer()

   def get_sent_detector(self):

       sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
       return sent_detector


   #get datasets from the csv--------
   #2：
   def get_train_data(self):

        train_data = pd.read_csv("/Users/xiaoyiwen/Desktop/Word2vec_Sentiment_Analysis/MasterProject/data_Preprocessing/Datasets/kaggle_data/labeledTrainData.tsv",header = 0, delimiter='\t', quoting=3)
        return train_data

   #3：
   def get_test_data(self):

        test_data = pd.read_csv("/Users/xiaoyiwen/Desktop/Word2vec_Sentiment_Analysis/MasterProject/data_Preprocessing/Datasets/kaggle_data/testData.tsv",header=0, delimiter='\t',quoting=3)
        return test_data

   #4：
   def get_unlabeled_data(self):


        unlabeled_data = pd.read_csv("/Users/xiaoyiwen/Desktop/Word2vec_Sentiment_Analysis/MasterProject/data_Preprocessing/Datasets/kaggle_data/unlabeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
        return unlabeled_data
   def get_polarity_data(self):
        polarity = pd.read_csv("/Users/xiaoyiwen/Desktop/MasterProject/MasterProject/data_Preprocessing/Datasets/kaggle_data/polarity.tsv", header=0, delimiter='\t', quoting=3)
        return polarity
   #5：
   # get all the review
   def get_clean_review_lists(self,review_datasets):
       clean_review_datasets = []
       for review in review_datasets:
           clean_review_datasets.append(Datasets.clean_text(review))
       return clean_review_datasets


   #clean the datasets------------
   #6：
   # preprocess the text and split each review to a list of tokens
   def clean_text(review):
       # 1: remove the tags
       review = BeautifulSoup(review).get_text()
       # 2.remove the non alpha
       text = re.sub("[^a-zA-Z]", " ", review)
       # 3.remove split the tokens
       lowercase = text.lower().split()
       # 4:remove the stopwords
       stops = set(stopwords.words("English"))
       words = [w for w in lowercase if not w in stops]

       return words

   def clean_text_without_filter_stopwords(review):
       # 1: remove the tags
       review = BeautifulSoup(review).get_text()
       # 2.remove the non alpha
       text = re.sub("[^a-zA-Z]", " ", review)
       # 3.remove split the tokens
       words = text.lower().split()

       return words


   def LDA_preprocessing( self, review,do_stem = False):
       # 1: remove the tags
       review = BeautifulSoup(review).get_text()
       # 2.remove the non alpha
       text = re.sub("[^a-zA-Z]", " ", review)
       # 3.remove split the tokens
       lowercase = text.lower().split()
       # 4:remove the stopwords
       stops = set(stopwords.words("English"))
       words = [w for w in lowercase if not w in stops]
       if(do_stem ==True):
           words = [Datasets.lemma.lemmatize(word) for word in words]
       return words

   #7：
   #split each review into sentences and clean the sentences -> to a list
   def transfer_review_to_sentences(review, sent_detector):

        #split the review to a list
        sentences = sent_detector.tokenize(review.strip())

        sentencs_list = []
        for sentence in sentences:
            if(len(sentence)>0):
             words = Datasets.clean_text_without_filter_stopwords(sentence)
             sentencs_list.append(words)

        return sentencs_list

   #8：
   def transfer_datasets_to_sentences(self,datasets,sent_detector):

       all_sentences = []

       for review in datasets:
           all_sentences += Datasets.transfer_review_to_sentences(review,sent_detector)

       return all_sentences




   #9：get averaged vector review
   def return_averaged_vector_review(dimension, review, model):

       review_vector = np.zeros((dimension,), dtype="float32")

       vocab = set(model.wv.index2word)

       num_words = 0

       for word in review:
           if (word in vocab):
               num_words += 1
               review_vector = np.add(review_vector, model[word])

       # divide the result
       review_vector = np.divide(review_vector, num_words)

       return review_vector


   #10：
   def return_total_vector(self, reviews, model, dimension):

       # should be 25000(train reviews) * 300(dimension)
       matrix = np.zeros((len(reviews), dimension), dtype="float32")

       count = 0

       for review in reviews:

           if (count % 1000 == 0):
               print("Review %d of %d" % (count, len(reviews)))

           matrix[count] = Datasets.return_averaged_vector_review(dimension, review, model)
           count = count + 1

       return matrix


   #11：
   # plot word cloud
   def plot_word_cloud(self, terms):
       text = terms.index
       text = ' '.join(list(text))

       # lower the max fontsize
       wordcloud = WordCloud(max_font_size=40).generate(text)
       plt.figure(figsize=(25, 25))
       plt.imshow(wordcloud, interpolation="bilinear")
       plt.axis("off")
       plt.show()




   # #11: build up a vocab
   # def build_up_vocab(self,dataset,vocab_path):
   #     #
   #     if(os.path.exists(vocab_path)):
   #         vocab = open(vocab_path,'rb')
   #         vocab = pickle.load(vocab)
   #         print("vocab successfully loaded!")
   #
   #     else:
   #         word_frequent = defaultdict(int)
   #
   #         for review in dataset:
   #             words = Datasets.clean_text_without_filter_stopwords(review)
   #             for word in words:
   #                 word_frequent[word] +=1
   #         print("loaded finished")
   #
   #         #create vocab
   #         vocab = {}
   #         i = 1
   #         vocab['UNKNOW_TOKEN'] = 0
   #         for word, freq in word_frequent.items():
   #             if freq>5:
   #                 vocab[word] =i
   #                 i +=1
   #
   #         #save the vocab
   #         with open(vocab_path,'wb') as file:
   #             pickle.dump(vocab,file)
   #             print(len(vocab))
   #             print("vocab save finished")
   #
   #     return vocab



#word2vec get the tokens for trainig
   def get_normalized_data(self,dataset_name):
    reviews = []
    review_sentences = []
    review_tokens = []


    data_main = Datasets()
    if(dataset_name =="train"):
        train_data = data_main.get_train_data()
        data = train_data
    if(dataset_name =="test"):
        test_data = data_main.get_test_data()
        data = test_data
    if(dataset_name =="unlabeled"):
        unlabeled_data = data_main.get_unlabeled_data()
        data = unlabeled_data
    if (dataset_name == "polarity"):
        data = data_main.get_polarity_data()

    print("here")
    # clean the test dataset
    for review in data["review"]:
        text = BeautifulSoup(review)
        cleaned_text = text.get_text().encode('ascii','ignore')
        cleaned_string = cleaned_text.decode('utf-8')
        cleaned_review  = data_main.clean_text_to_text(cleaned_string)
        reviews.append(cleaned_review)
        sentence = tokenize.sent_tokenize(cleaned_review)
        # number of the review
        review_sentences.append(sentence)

        for s in sentence:
           if(len(s)>0):
               tokens = text_to_word_sequence(s)
               #filter out non-alpha
               tokens = [token for token in tokens if token.isalpha()]
               #filter out those short letters
               tokens = [t for t in tokens if len(t)>1]
               review_tokens.append(tokens)


    return reviews, review_sentences, review_tokens

   def clean_text_to_text(self,review):
       #remove the the',",\
       review = re.sub(r"\\", "", review)
       review = re.sub(r"\'", "", review)
       review = re.sub(r"\"", "", review)
       #return the lower case
       text = review.strip().lower()

       return text