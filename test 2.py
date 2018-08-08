from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt




# visualize the word2vec
def tsne_plot(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca',n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16,16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i],y[i]),xytest = (5,2), textcoords = 'offset points',ha = 'right',va = 'bottom')

    plt.show()



modelname = "./word2vec_model/300features_10min_10window_NEG10"
model = Word2Vec.load(modelname)

tsne_plot(model)
