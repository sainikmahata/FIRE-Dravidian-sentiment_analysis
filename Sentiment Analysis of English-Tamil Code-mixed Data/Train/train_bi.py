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


word_inp=np.load('word_inp.npy')
print (word_inp.shape)
ln_inp=np.load('ln_inp.npy')
print (ln_inp.shape)
labels_out=np.load('label_out.npy')
print (labels_out.shape)


train_x1=word_inp[:12097]
train_x2=ln_inp[:12097]
train_y=labels_out[:12097]

test_x1=word_inp[12097:]
test_x2=ln_inp[12097:]
test_y=labels_out[12097:]

#manage class imbalances
array=[i[0] for i in train_y.tolist()]
class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(array),
                                                array)
class_weights_dict={}
for i,j in enumerate(class_weights):
   class_weights_dict[i]=j

#load the fasttext embedding
embeddings_index = dict()
f = open('vectors_f.txt','r',errors='ignore',encoding='utf-8')
for line in f:
   values = line.split()
   word = values[0]
   coefs = asarray(values[1:], dtype='float32')
   embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((np.amax(word_inp)+ 1, 100))
for word, i in dictio.items():
   embedding_vector = embeddings_index.get(word)
   if embedding_vector is not None:
       embedding_matrix[i] = embedding_vector

#create the model
input1=Input(shape=(100,))
emb=Embedding(np.amax(word_inp)+ 1,
                100,
                weights=[embedding_matrix],
                trainable=False, 
                mask_zero=True)(input)
input2=Input(shape=(100,))
emb2=Embedding(np.amax(ln_inp)+ 1,100,mask_zero=True)(input2)
con=tensorflow.keras.layers.Concatenate(axis=-1)([emb,emb2])
lstm1=Bidirectional(LSTM(64),return_sequences=True)(emb)
lstm2=Bidirectional(LSTM(64))(lstm1)
output=Dense(5, activation='softmax')(lstm2) #no. of labels + 1
model=Model([input1,input2], output)
sparse=tensorflow.keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=sparse, optimizer='adam', metrics=['accuracy'])
print(model.summary())
es=tensorflow.keras.callbacks.EarlyStopping(
   monitor='val_loss', patience=5, mode='min')
mc=tensorflow.keras.callbacks.ModelCheckpoint(
   'checkpoint_bi.h5', monitor='val_loss', verbose=1, save_best_only=True,
   mode='auto')
model.fit([train_x1,train_x2], train_y, epochs=50, batch_size=32, validation_split=0.1,class_weight=class_weights_dict, callbacks=[es,mc])
model.save('model_bi.h5')

#load the checkpoint to claculate accuracy and generate the classification report
model=load_model('checkpoint_bi.h5')
loss, accuracy= model.evaluate([test_x1,test_x2], test_y)
print ('Accuracy: %f',accuracy)
prediction=model.predict([test_x1,test_x2])
y_pred=[]
for i in prediction:
    y_pred=np.argmax(i)
y_true=test_y.flatten()
report=classification_report(y_true,y_pred)
print (report)

