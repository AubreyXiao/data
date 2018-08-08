from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

modelname = "./word2vec_model/300features_10min_20window_NEG10"

#load the model

model = Word2Vec.load(modelname)

#get the vocab
X = model[model.wv.vocab]

#2 dimension
pca = PCA(n_components=2)

#
results = pca.fit_transform(X)

#create a scatter plot
plt.scatter(results[:,0],results[:,1])

words = list(model.wv.vocab)

for i, word in enumerate(words):

    plt.annotate(word, xy =(results[i,0],results[i,1]))

plt.show()
