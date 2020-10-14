import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import json
import tensorflow
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Bidirectional
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

dictio=json.load(open('word_dict.json','r'))


word_inp=np.load('test_word_inp.npy')
print (word_inp.shape)
ln_inp=np.load('test_ln_inp.npy')
print (ln_inp.shape)

f=open('test_tokenized.txt','r',errors='ignore').read().split('\n')
f1=open('result.txt','w')
model=load_model('checkpoint_bi.h5')
prediction=model.predict([word_inp,ln_inp])
for i in range(len(f)):
    if np.argmax(prediction[i])==0:
        f1.write(f[i]+'\tPositive\n')
    elif np.argmax(prediction[i])==1:
        f1.write(f[i]+'\tNegative\n')
    elif np.argmax(prediction[i])==2:
        f1.write(f[i]+'\tMixed_feelings\n')
    elif np.argmax(prediction[i])==3:
        f1.write(f[i]+'\tunknown_state\n')
    else:
        f1.write(f[i]+'\tnot_Tamil\n')
f1.close()

