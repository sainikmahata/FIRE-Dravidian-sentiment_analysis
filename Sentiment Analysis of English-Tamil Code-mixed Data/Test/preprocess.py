import keras
import tensorflow
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from tqdm import tqdm

word_dict=json.load(open('word_dict.json','r'))

f=open('test_tokenized.txt','r',errors='ignore').read().split('\n')
l1=[]
for i in range(len(f)):
    l=[]
    tokens=f[i].split()
    for j in tokens:
        if j in word_dict.keys():
            l.append(int(word_dict[j]))
    l1.append(l)
pad=pad_sequences(l1,maxlen=100,padding='post')
test_word_inp=np.array(pad)
print (test_word_inp.shape)
np.save('test_word_inp.npy',test_word_inp)

##for 2nd input (english and non-english words)##
import nltk
from nltk.corpus import words
setofwords=set(words.words())
f=open('test_tokenized.txt','r',encoding='utf-8').read().split('\n')[:-1]
l1=[]
for i in f:
    l=[]
    tokens=i.split()
    for token in tokens:
        if token in setofwords:
            l.append(int(5)) #check dict_assumtions.txt for one-hot assumptions
        else:
            l.append(int(6))
    l1.append(l)
pad=pad_sequences(l1,maxlen=100, padding='post')
print (pad.shape)
np.save('test_ln_inp.npy',pad)
##################################################
