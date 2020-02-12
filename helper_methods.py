"""
Helper methods
"""

import spacy
import pandas as pd
import subprocess
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

def get_different_words(sent_tuple, parser, return_dep=False):

    """
    Method to get different words
    Uses spacy tokenizer instead of bert
    Takes a tuple of string sentences and a parser

    Returns a dict with different words as keys, index of word in sentence as value
    alternatively the dep tags of the different words if return_dep=True
    
    """
    sen_0 = parser(sent_tuple[0])
    sen_1 = parser(sent_tuple[1])

    length_0 = len(sen_0)
    length_1 = len(sen_1)

    differ_dict_0 = {}
    differ_dict_1 = {}

    index_of_word = 0

    for token in sen_0:
        word = token.text
        dep_tag = token.dep_
        if not word in sen_1.text:
            if return_dep:
                differ_dict_0[word] = dep_tag
            else:
                differ_dict_0[word] = index_of_word
            
            index_of_word += 1
        else:
            index_of_word += 1

    index_of_word = 0
    for token in sen_1:
        word = token.text
        dep_tag = token.dep_
        if not word in sen_0.text:
            if return_dep:
                differ_dict_1[word] = dep_tag
            else:
                differ_dict_1[word] = index_of_word
            
            index_of_word += 1
        else:
            index_of_word += 1


    differ_dict = (differ_dict_0, differ_dict_1)
    
    return differ_dict

def get_parser(model_name):
    """
    Helper-Method to get spacy parser
    takes a name of language model (?) as string
    returns parser
    """
    try:
        parser = spacy.load(model_name)
        return parser
    except:
        bashCommand = "python3 -m spacy download " + model_name
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        try:
            parser = spacy.load(model_name)
            return parser
        except:
            print('Something went horribly wrong. Check your model')


def compute_similarity(dependents):

    similarity = 0
    cur, prev = None, None
    for w in dependents:
        #cur = parser(w) #only for testing
        cur = w
        #print(cur)
        if prev == None:
            prev = cur
            continue
        #print(prev)
        similarity += prev.similarity(cur)
        #print(similarity)
        prev = cur
    return similarity



def compute_similarity_BERT(dependents):

    similarity = 0
    cur, prev = None, None
    i = 0
    for w in dependents:
        #cur = parser(w) #only for testing
        cur = np.reshape(w, (1, 768))
        #print(cur.shape)
        if i == 0:
            prev = cur
            i+=1
            continue
        #print(prev)
        similarity += cosine_distances(prev, cur)
        #print(similarity)
        prev = cur
    return similarity

