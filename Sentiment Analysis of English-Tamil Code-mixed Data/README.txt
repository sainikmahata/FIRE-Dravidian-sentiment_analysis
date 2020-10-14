##For Training##
1. change current directory to "Train"
2. run the "dict_creation.py" program to create one-hot word dictionary.
3. run the "preprocess.py" program to create the word and language tag input numpy files and label output numpy file.
4. run the "train_bi.py" program to train the model. the model uses fastext embeddings, that is stored in "vectors_f.txt" file.
5. After the training is completed, the model will output a "checkpoint_bi.h5" and "model.h5" files. we will use the checkpoint file for the testing purpose.

##For Testing##
1. the test data was tokenized and stored in 'test_tokenized.txt'.
2. Run the 'preprocess.py' file to generate the input numpy arrays.
3. run the 'test.py' program to generate the outputs.
4. the output is stored in 'result.txt' 