from ampligraph.latent_features import TransE
from ampligraph.latent_features import HolE
from ampligraph.utils import save_model, restore_model
import numpy as np
import _pickle as cPickle
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen
import os
import _pickle as cPickle
import csv
from itertools import chain

with open('diginetica/all_train_seq.txt','rb')as f:
    all_train=cPickle.load(f)


triples =[]
for i in range (len(all_train)):
    for j in range(len(all_train[i])):
        for k in range(j+1, len(all_train[i])):
            #if k-j < 10:
                triples.append([str(all_train[i][j]),str(i),str(all_train[i][k])])
    
with open('KG_triples.csv', 'w') as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerows(triples)
        
X=load_from_csv('.', 'KG_triples.csv', sep=',')
model = TransE(batches_count=100,
                epochs=50,
                k=200,
                eta=30,
                embedding_model_params={'corrupt_sides': ['s','o'], 'negative_corruption_entities': 'all', 'norm': 1, 'normalize_ent_emb': False},
                loss = 'multiclass_nll',
                optimizer='adam',
                optimizer_params={'lr':1e-3},
                regularizer='LP',
                regularizer_params={'p':3, 'lambda':1e-5},
                seed=0,
                verbose=True)


model.fit(X)

vocab = list(set(chain(*all_train)))
vocab = [str(each) for each in vocab]
print(len(vocab))

item_embeddings = dict(zip(vocab, model.get_embeddings(vocab)))
with open("diginetica/transe_emb","wb") as f:
        cPickle.dump(item_embeddings,f)

