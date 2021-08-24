import json
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk


##########################################
############## Data Loading ##############
##########################################

reviews = [] # empty array of dicts to store data
with open('apps_cleaned.json') as file:  # open cleaned json file
    for reviewjson in file: # read through lines
        review = json.loads(reviewjson) # load json object
        reviews.append(review) # append dict to array of dict


##########################################
##############    Models    ##############
##########################################

model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
split_sentence = nltk.data.load('tokenizers/punkt/english.pickle') # split sentence tokenizer

##########################################
############# Text Processing ############
##########################################

sentences = ['This framework generates embeddings for each input sentence but I am not sure how long this sentence is.... Sentences are passed as a list of string. The quick brown fox jumps over the lazy dog.']

def Preprocess(text):
    new_text = "" # new string
    i = 0
    while i < len(text):
        if text[i] == ".": # truncates periods. Given "a....", returns "a. "
            while i < len(text) - 1:
                if text[i + 1] == ".":
                    i += 1
                else:
                    new_text += ". "
                    break
        elif text[i] == "!" or text[i] == "?": # adds empty space after string of exclamation. Given "a!!!!", returns "a!!!! "
            while i < len(text) - 1:
                new_text += text[i]
                if text[i + 1] != "!" and text[i + 1] != "?":
                    new_text += " "
                    break
                i += 1
        else:
            new_text += text[i] # adds to new string
        i += 1
    return new_text

def batch_embedding(text): # given a review, return an embedding in 1 * 768 an array (sentence splitted)
    processed_text = Preprocess(text) # preprocess text to deal with splitting problems
    batch_sentences = split_sentence.tokenize(processed_text) # lists of sentences
    embeddings = model.encode(batch_sentences) # batch sentence into batch embeddings
    mean_embeddings = np.mean(embeddings, axis = 0)
    return mean_embeddings # return embedding in np array

def block_embedding(text): # only 128 max word pieces, probably not enough
    embeddings = model.encode(text) # sentence into embeddings
    return embeddings # return embedding in np array
 
##########################################
##############  Embedding  ###############
##########################################

train_x_batch = np.empty((1,768)) # store training data for block
train_y = np.empty((1)) # store training data
test_x_batch = np.empty((1,768)) # store training data for batch
test_y = np.empty((1)) # store testing data

sample_index = np.random.choice(range(0,len(reviews)), size = 2000, replace = False) # empty array of int to store training indices

for i in range(0,len(sample_index)):
    print(i)
    if i < 1000: # training data
        if reviews[sample_index[i]]['overall'] > 3: # rating == 4 or 5
            train_y = np.append(train_y, 1) # append high
        else: # rating == 1 or 2
            train_y = np.append(train_y, 0) # append low
        train_x_batch = np.append(train_x_batch, np.array([batch_embedding(reviews[sample_index[i]]['reviewText'])]), axis = 0) # append batch embedding
    else:
        if reviews[sample_index[i]]['overall'] > 3: # rating == 4 or 5
            test_y = np.append(test_y, 1) # append high
        else: # rating == 1 or 2
            test_y = np.append(test_y, 0) # append low
        test_x_batch = np.append(test_x_batch, np.array([batch_embedding(reviews[sample_index[i]]['reviewText'])]), axis = 0) # append batch embedding


train_x_batch = np.delete(train_x_batch, 0, axis = 0) # delete random data
train_y = np.delete(train_y, 0) # delete random data
test_x_batch = np.delete(test_x_batch, 0, axis = 0) # delete random data
test_y = np.delete(test_y, 0) # delete random data

np.savetxt('s_train_x_batch.txt',train_x_batch)
np.savetxt('s_train_y.txt',train_y)
np.savetxt('s_test_x_batch.txt',test_x_batch)
np.savetxt('s_test_y.txt',test_y)

