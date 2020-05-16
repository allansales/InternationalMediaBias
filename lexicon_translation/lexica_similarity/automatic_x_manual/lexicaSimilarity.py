from gensim.models import KeyedVectors
import sys
import numpy as np
import pandas as pd

path = sys.argv[1] # path to embeddings .bin

model = KeyedVectors.load_word2vec_format(path, binary=True)

automatic_trans_lex = pd.read_csv("./lexicons_pons.csv")

manual_trans_lex = pd.read_csv("./manually_translated_lexicons.csv")
manual_trans_lex.at[0,'lex']="modalization"
manual_trans_lex.at[1,'lex']="pressuposition"

# WMD Between Lexica


def generate_rand_lex(lex_name, lex, n_samples=5):
    lex_len = len(lex.split())

    # generate random lexica
    random_lex = [" ".join(random.sample(model.wv.vocab.keys(), lex_len)) for i in range(n_samples)]
    rand_lex_df = pd.DataFrame(random_lex, columns=["rand_words"])

    # assign random label to each lexicon
    lex = pd.DataFrame(np.tile([lex],n_samples), columns=["automatic_words"])
    lexica_df = pd.concat([lex,rand_lex_df], axis=1)

    # compute wmd between the lexica
    wmd = lexica_df.apply(lambda x: model.wmdistance(x.automatic_words.split(), x.rand_words.split()), axis=1)
    return(wmd.mean())


# Generate Most Similar Lexicon
def most_similar_lex(lex):
    lex = lex.split()
    most_similar_lex = []
    for word in lex:
        if word in model.vocab:
            most_similar_word = model.most_similar(word, topn=1)[0][0] #get the word, not the entire tuple
            most_similar_lex.append(most_similar_word)
    return(most_similar_lex)

f = open("lexica_dist_semantic.csv", "w")
f.write("lex,type_1,type_2,dist"+"\n")

mod_most_sim_lex = most_similar_lex(manual_trans_lex.words[0])
pre_most_sim_lex = most_similar_lex(manual_trans_lex.words[1])

# Auto x Random
import random
mod_auto_rand_dist = generate_rand_lex(manual_trans_lex.lex[0], manual_trans_lex.words[0])
pre_auto_rand_dist = generate_rand_lex(manual_trans_lex.lex[1], manual_trans_lex.words[1])
f.write("mod,auto,random,"+str(mod_auto_rand_dist)+"\n")
f.write("pre,auto,random,"+str(pre_auto_rand_dist)+"\n")


# Auto x Manual
auto_man_df = automatic_trans_lex.merge(manual_trans_lex, how = "inner", on=["lang","lex"])
auto_man_dist = auto_man_df.apply(lambda x: model.wmdistance(x.words_x.split(), x.words_y.split()), axis=1)
f.write("mod,auto,manual,"+str(auto_man_dist[0])+"\n")
f.write("pre,auto,manual,"+str(auto_man_dist[1])+"\n")

# Auto x Most Similar to manual
mod_auto_sim_dist = model.wmdistance(auto_man_df.words_x[0], mod_most_sim_lex)
pre_auto_sim_dist = model.wmdistance(auto_man_df.words_x[1], pre_most_sim_lex)
f.write("mod,auto,sim_to_manual,"+str(mod_auto_sim_dist)+"\n")
f.write("pre,auto,sim_to_manual,"+str(pre_auto_sim_dist)+"\n")

# Manual x Most Similar to manual
mod_manual_sim_dist = model.wmdistance(auto_man_df.words_y[0], mod_most_sim_lex)
pre_manual_sim_dist = model.wmdistance(auto_man_df.words_y[1], pre_most_sim_lex)
f.write("mod,manual,sim_to_manual,"+str(mod_manual_sim_dist)+"\n")
f.write("pre,manual,sim_to_manual,"+str(pre_manual_sim_dist))
f.close()

# Sintatic Similarity between Lexica
def n_similar_items(a, b) :
    not_similar = [item for item in a if item not in b ]
    n_diff = len(not_similar)
    n_similar = len(a) - n_diff
    return(n_similar, n_diff, not_similar)

sin_dist = auto_man_df.apply(lambda x: n_similar_items(x.words_x.split(), x.words_y.split()), axis=1)

f = open("lexica_dist_sintatic.csv", "w")
f.write("lex,type_1,type_2,n_in_common,n_distinct,distinct_words"+"\n")
f.write("mod,auto,man,"+str(sin_dist[0][0])+","+str(sin_dist[0][1])+","+str(sin_dist[0][2])+"\n")
f.write("pre,auto,man,"+str(sin_dist[1][0])+","+str(sin_dist[1][1])+","+str(sin_dist[1][2])+"\n")

### WMD Change by word
#How WMD changes when we replace one word from the manual lexicon with a random word

def replace_and_compute_wmd(lex, idx, word): # replace one word and compute WMD
    new_lex = list(lex)
    new_lex[idx] = word
    return(model.wmdistance(new_lex, lex))

def generate_rand_word(model):
    return(random.sample(model.wv.vocab.keys(), 1)[0])

def generate_sim_word(word, model):
    return(model.most_similar(word, topn=1)[0][0]) #get the word, not the entire

NUM_GENERATED_WORDS = 50

lex_dists = []
for lex_idx in range(manual_trans_lex.shape[0]):
    lex = manual_trans_lex.words[lex_idx].split()
    dists = []
    for i, word in enumerate(lex):
        if word in model.vocab:
            new_word = generate_sim_word(word, model)
            dist = [replace_and_compute_wmd(lex, idx, new_word) for idx, item in enumerate(lex)]
    #for i in range(NUM_GENERATED_WORDS):
        #rand_word = generate_rand_word(model)
        #dist = [replace_and_compute_wmd(lex, idx, rand_word) for idx, item in enumerate(lex)]
            dists.append(dist)
    lex_dists.append(dists)

def semantic_agreement_level(n_words, one_word_dist, total_dist):
    semantic_equal = n_words-(total_dist/one_word_dist)
    agreement = (semantic_equal)/n_words
    return(agreement)


mean_dists = [np.mean(dists) for dists in lex_dists]
lex_sizes = auto_man_df.words_y.apply(lambda x: len(x.split()))

semantic_agreement = semantic_agreement_level(lex_sizes, mean_dists, auto_man_dist)

f.write("mod,manual,manual_with_single_word_change,"+str(semantic_agreement[0])+"\n")
f.write("pre,manual,manual_with_single_word_change,"+str(semantic_agreement[1]))
f.close()

