from gensim.models import KeyedVectors
import sys
import numpy as np
import pandas as pd
import random

path = sys.argv[1] # path to embeddings .bin
n_runs = sys.argv[2] # number of times we generate random lex

model = KeyedVectors.load_word2vec_format(path, binary=True)

automatic_trans_lex = pd.read_csv("./lexicons_pons.csv")
automatic_trans_lex = automatic_trans_lex[automatic_trans_lex.lang == "eng"]

manual_trans_lex = pd.read_csv("./manually_translated_lexicons.csv")
manual_trans_lex.at[0,'lex']="modalization"
manual_trans_lex.at[1,'lex']="pressuposition"

# Auto X Manual dist
auto_man_df = automatic_trans_lex.merge(manual_trans_lex, how = "inner", on=["lang","lex"])
auto_man_dist = auto_man_df.apply(lambda x: model.wmdistance(x.words_x.split(), x.words_y.split()), axis=1)

# Manual x Random dist
## Random random words 
def generate_rand_word(model, lex):
    rand_lex = lex.apply(lambda x: random.sample(model.wv.vocab.keys(), len(x.split())))
    return(rand_lex)

## distance computation
dists = []
for i in range(n_runs):
    rand_lex = generate_rand_word(model, manual_trans_lex.words)
    manual_rand_dist = [model.wmdistance(manual.split(), random) for manual, random in zip(manual_trans_lex.words, rand_lex)]
    dists.append(manual_rand_dist)

# Normalization Step
dists = np.array(dists)
mean_dist = np.mean(dists, axis=0)
sim = 1-(auto_man_dist/(mean_dist))

info_df = pd.DataFrame({"lex":manual_trans_lex.lex,"auto_man":auto_man_dist, "mean_dist":mean_dist, "similarity":sim})
info_df.to_csv("sim_df.csv")