import json
import numpy as np
from transformers import BertTokenizer, BertTokenizerFast, BertModel, AutoModel, AutoTokenizer
import torch
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

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased') # bert tokenizer
model = BertModel.from_pretrained('bert-base-cased',output_hidden_states = True) # bert model
# tokenizer2 = AutoTokenizer.from_pretrained('bert-base-cased')
# model2 = AutoModel.from_pretrained('bert-base-cased')

split_sentence = nltk.data.load('tokenizers/punkt/english.pickle') # split sentence tokenizer


##########################################
############# Text Processing ############
##########################################

# text = "This is a sample sentence.....I wonder if it splits!!!!!guess it doesn't.'I wonder'' if Dr. Green's car works?"

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


##########################################
##############  Embedding  ###############
##########################################

# def bert_manual(text): # tokenizer for one sentence
#     marked_text = "[CLS]" + text + " [SEP]" # add class and seperation token
#     tokenized_text = tokenizer.tokenize(marked_text) # tokenization
#     indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # map to indices
#     segment_ids = [1] * len(tokenized_text) # segment ids (all 1 sentence)
#     return (torch.tensor([indexed_tokens]), torch.tensor([segment_ids])) # return Pytorch tensors of inputs

def batch_embedding(text): # given a review, return an embedding in 1 * 768 an array (sentence splitted)
    processed_text = Preprocess(text) # preprocess text to deal with splitting problems
    batch_sentences = split_sentence.tokenize(processed_text) # lists of sentences
    inputs = tokenizer(batch_sentences, padding = True, truncation = True, return_tensors = "pt") # tokenize the batch
    outputs = model(**inputs) # run bert
    token_vecs = outputs[2][-2] # get all hidden layers and then select second to last hidden layer
    # print(token_vecs.size())
    review_embedding = torch.mean(token_vecs, dim = [0,1]) # average by token and then sentence
    return (review_embedding.detach().numpy()) # return embedding in np array

def block_embedding(text): # given a review, return an embedding in an 1 * 768 array (whole review)
    inputs = tokenizer(text, truncation = True, return_tensors = "pt") # tokenize review
    outputs = model(**inputs) # run bert
    token_vecs = outputs[2][-2][0] # get all hidden layers and then select second to last hidden layer and first batch
    # print(token_vecs.size())
    review_embedding = torch.mean(token_vecs, dim = 0) # average by token
    return (review_embedding.detach().numpy())

train_x_block = np.empty((1,768)) # store training data for block
train_x_batch = np.empty((1,768)) # store training data for batch
train_y = np.empty((1)) # store training data
test_x_block = np.empty((1,768)) # store testing data
test_x_batch = np.empty((1,768)) # store training data for batch
test_y = np.empty((1)) # store testing data

sample_index = np.random.choice(range(0,len(reviews)), size = 30000, replace = False) # empty array of int to store training indices

for i in range(0,len(sample_index)):
    print(i)
    if i < 10000: # training data
        if reviews[sample_index[i]]['overall'] > 3: # rating == 4 or 5
            train_y = np.append(train_y, 1) # append high
        else: # rating == 1 or 2
            train_y = np.append(train_y, 0) # append low
        train_x_block = np.append(train_x_block, np.array([block_embedding(reviews[sample_index[i]]['reviewText'])]), axis = 0) # append block embedding
        train_x_batch = np.append(train_x_batch, np.array([batch_embedding(reviews[sample_index[i]]['reviewText'])]), axis = 0) # append batch embedding
    else:
        if reviews[sample_index[i]]['overall'] > 3: # rating == 4 or 5
            test_y = np.append(test_y, 1) # append high
        else: # rating == 1 or 2
            test_y = np.append(test_y, 0) # append low
        test_x_block = np.append(test_x_block, np.array([block_embedding(reviews[sample_index[i]]['reviewText'])]), axis = 0) # append block embedding
        test_x_batch = np.append(test_x_batch, np.array([batch_embedding(reviews[sample_index[i]]['reviewText'])]), axis = 0) # append batch embedding


train_x_block = np.delete(train_x_block, 0, axis = 0) # delete random data
train_x_batch = np.delete(train_x_batch, 0, axis = 0) # delete random data
train_y = np.delete(train_y, 0) # delete random data
test_x_block = np.delete(test_x_block, 0, axis = 0) # delete random data
test_x_batch = np.delete(test_x_batch, 0, axis = 0) # delete random data
test_y = np.delete(test_y, 0) # delete random data

np.savetxt('train_x_block.txt',train_x_block)
np.savetxt('train_x_batch.txt',train_x_batch)
np.savetxt('train_y.txt',train_y)
np.savetxt('test_x_block.txt',test_x_block)
np.savetxt('test_x_batch.txt',test_x_batch)
np.savetxt('test_y.txt',test_y)

