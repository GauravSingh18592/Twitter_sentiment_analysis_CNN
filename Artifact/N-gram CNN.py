import os
path_ = 'C:/Users/User/Downloads'
os.chdir(path_)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


tweet_feb = pd.read_excel("tweet_gnew.xlsx", sheet_name='s1', header=1)
tweet_df = pd.DataFrame(data=tweet_feb, columns=['Date', 'Tweet Text', 'Followers', 'Follows', 'Retweets', 'target'])
tweet_df.rename(columns={"Tweet Text": "Tweet_Text"}, inplace=True)

# **************************** DATA PREPRATION ******************************
# TEST SAMPLE DATA
main_df = tweet_df.copy()
main_df.drop(columns=['Date', 'Followers', 'Follows', 'Retweets'], axis=1, inplace=True)
main_df = main_df.dropna()
main_df.reset_index(drop=True, inplace=True)
main_df['target'] = pd.factorize(main_df['target'])[0]

# EVALUTION_Unknown Sample data
main_copy_df = tweet_df.copy()
main_copy_df.drop(columns=['Date', 'Followers', 'Follows', 'Retweets'], axis=1, inplace=True)
main_copy_df = main_copy_df.fillna("none")
main_copy_df = main_copy_df.loc[main_copy_df["target"] == 'none']



# *************Creating**Super set***and training set************************************************************
tweet_super_set_test = main_copy_df.Tweet_Text  # to be used for prediction

tweet_text = main_df.Tweet_Text  # to be used for Training
target = main_df.target  # to be used for training



# ***********************
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

def labelize_tweets_ug(tweet_text, targets):
    result = []
    prefix = targets
    for i, t in zip(tweet_text.index, tweet_text):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result
cores = multiprocessing.cpu_count()
# ********************** Labelizing, Vocab creation for treating Tweet Text ************
all_tweet_w2v = labelize_tweets_ug(tweet_text, 'all')



# _____________________________word vectorizing using cbow ______________________________
model_unigram_cbow = Word2Vec(sg=0, size=100, negative=10, window=2,
                         min_count=2, workers=cores, alpha=0.065,
                         min_alpha=0.065)  # x values
model_unigram_cbow.build_vocab([x.words for x in tqdm(all_tweet_w2v)])  # vocab creation x
for epoch in range(50):
    model_unigram_cbow.train(utils.shuffle([x.words for x in tqdm(all_tweet_w2v)]),
                        total_examples=len(all_tweet_w2v), epochs=1)
    model_unigram_cbow.alpha -= 0.002
    model_unigram_cbow.min_alpha = model_unigram_cbow.alpha
### ___________________word vectorizing using_skipgram __________________
model_unigram_skipgram = Word2Vec(sg=1, size=100, negative=10, window=2, min_count=2,
                       workers=cores, alpha=0.065, min_alpha=0.065)
model_unigram_skipgram.build_vocab([x.words for x in tqdm(all_tweet_w2v)])
for epoch in range(50):
    model_unigram_skipgram.train(utils.shuffle([x.words for x in tqdm(all_tweet_w2v)]),
                      total_examples=len(all_tweet_w2v), epochs=1)
    model_unigram_skipgram.alpha -= 0.002
    model_unigram_skipgram.min_alpha = model_unigram_skipgram.alpha
#ADD cosine similarity and tsne for mapped vectors.
# ------------- Saving the Word to Vector model ------------------
'''****************************************
Word to Vector converts each word to a unique integer value in wv.vocab
and vectors are contained in wv vectors
*****************************************'''
model_unigram_cbow.save('w2v_model_unigram_cbow.word2vec')
model_unigram_skipgram.save('w2v_model_unigram_skipgram.word2vec')
# ----------------------------------------------------------------

# ------------- Loading Word-Vector Model ------------------------
from gensim.models import KeyedVectors

model_unigram_cbow = KeyedVectors.load(r'w2v_model_unigram_cbow.word2vec')
model_unigram_skipgram = KeyedVectors.load(r'w2v_model_unigram_skipgram.word2vec')
# ----------------------------------------------------------------

# -------------- Embedding index creation for Training Tweets Text -------------------
embeddings_index = {}
for w in model_unigram_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_unigram_cbow.wv[w], model_unigram_skipgram.wv[w])
print('Found %s word vectors.' % len(embeddings_index))
# --------------------------------------------------------------------------
# ________________________________ CNN KERAS NETWORK for ________________
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=90000)
tokenizer.fit_on_texts(tweet_text)
sequences = tokenizer.texts_to_sequences(tweet_text)

length = []
for x_ in tweet_text:
    length.append(len(x_.split()))
max(length)

tweet_text_seq = pad_sequences(sequences, maxlen=200)
print('Shape of data tensor:', tweet_text_seq.shape)

num_words = 90000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
#-------unit check---
np.array_equal(embedding_matrix[6], embeddings_index.get('Climate'))

seed = 7

# remember X_train_seq = x_seq

# --------------------------- KERAS NETWORK ---------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
# ------------------------ CNN model with N-gram ------------------------------------------------
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
early_stop2 = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
tweet_input = Input(shape=(200,), dtype='int32')
tweet_encoder = Embedding(90000, 200, weights=[embedding_matrix], input_length=200, trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=30, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=30, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=30, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)
merged = Dense(50, activation='relu')(merged)
merged = Dense(25, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(3)(merged)
output = Activation('softmax')(merged)
cnn_model = Model(inputs=[tweet_input], outputs=[output])
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.summary()

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report

#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(tweet_text_seq, target, test_size= 0.2, random_state = 24)

# ************---------- fit model CNN---------------------*********************
batch_size = 40
cnn_model.fit(X_train, Y_train, epochs=100, verbose=1, batch_size=batch_size,validation_data= (X_test, Y_test),callbacks=[early_stop2])

# ------------------------------- ONE HOT ENCODING ------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(Y_test)
cnn_model_y_test = encoder.transform(Y_test)
cnn_model_y_test = np_utils.to_categorical(cnn_model_y_test)
# ____________________________ PREDICTING O TEST TEST _____________________________
#analyze the results

cnn_model_y_pred = cnn_model.predict(X_test)
#cnn_model_y_prob = cnn_model.predict_proba(X_test)

cnn_model_score, cnn_model_acc = cnn_model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("score", cnn_model_score)
print("Accuracy", cnn_model_acc)
# ___________________________________________________________________________________
print(classification_report(Y_test, np.argmax(cnn_model_y_pred, axis = 1)))

#----------------------------- F1 SCORE CALCULATION ---------------------------------------------------#
from scipy import interp
from sklearn import metrics
from sklearn.metrics import f1_score

def cat_pred(nda):
    lst=[]
    for i in range(len(nda)):
        #print(nda[i,0], '\t', nda[i,1])
        if nda[i, 0] > nda[i, 1] and nda[i, 0] > nda[i, 2]:
            lst.append(0)
        elif nda[i, 1] > nda[i, 0] and nda[i, 1] > nda[i, 2]:
            lst.append(1)
        elif nda[i, 2] > nda[i, 0] and nda[i, 2] > nda[i, 1]:
            lst.append(2)
    return lst


#cnn_model_y_prob_F1 = cat_pred(cnn_model_y_prob)
cnn_model_y_pred_F1 = cat_pred(cnn_model_y_pred)
#cnn_model_f1_score = f1_score(Y_test, cnn_model_y_prob_F1, average='macro')
cnn_model_f1_micro_ovo = f1_score(Y_test, cnn_model_y_pred_F1, average='micro')

#-----------------------Im Able to print this one ------------------------------------------------------------#

#macro_roc_auc_ovo = roc_auc_score(cnn_model_y_test, cnn_model_y_prob, multi_class="ovo", average="macro")
macro_roc_auc_ovr = roc_auc_score(cnn_model_y_test, cnn_model_y_pred, multi_class="ovr", average="macro")
#weighted_roc_auc_ovo = roc_auc_score(cnn_03_y_test, cnn_03_y_prob, multi_class="ovo", average="weighted")
weighted_roc_auc_ovr = roc_auc_score(cnn_model_y_test, cnn_model_y_pred, multi_class="ovr", average="weighted")
'''
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
'''
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))


#-----------------------Add a crossvaildator to roc plot ------------------------------------------------------------#
'''
xc = cnn_model(kernel='linear', probability=True,
                     random_state=random_state)

from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import
cv = StratifiedKFold(n_splits=6)
from sklearn.model_selection import cross_val_score

lasso = cnn_model.fit()
print(cross_val_score(lasso, X_test, Y_test, cv=3))
'''
#Confusion matrix
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(X_test,Y_test,normalize='all'))
print(confusion_matrix(Y_test, np.argmax(cnn_model_y_pred, axis = 1)))
#Cohen_kappa_score
from sklearn.metrics import cohen_kappa_score
print(cohen_kappa_score(Y_test, np.argmax(cnn_model_y_pred, axis = 1)))
#hamming loss
from sklearn.metrics import hamming_loss
print(hamming_loss(Y_test, np.argmax(cnn_model_y_pred, axis = 1)))
# ----------------- ROC curve -------------------------------------------
# Compute ROC curve and ROC area for each class
cnn_model_fpr = dict()
cnn_model_tpr = dict()
cnn_model_roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    cnn_model_fpr[i], cnn_model_tpr[i], _ = roc_curve(cnn_model_y_test[:, i], cnn_model_y_pred[:, i])
    cnn_model_roc_auc[i] = metrics.auc(cnn_model_fpr[i], cnn_model_tpr[i])

# Compute micro-average ROC curve and ROC area
cnn_model_fpr["micro"], cnn_model_tpr["micro"], _ = roc_curve(cnn_model_y_test.ravel(), cnn_model_y_pred.ravel())
cnn_model_roc_auc["micro"] = metrics.auc(cnn_model_fpr["micro"], cnn_model_tpr["micro"])


# First aggregate all false positive rates
cnn_model_all_fpr = np.unique(np.concatenate([cnn_model_fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
cnn_model_mean_tpr = np.zeros_like(cnn_model_all_fpr)
for i in range(n_classes):
    cnn_model_mean_tpr += interp(cnn_model_all_fpr, cnn_model_fpr[i], cnn_model_tpr[i])

# Finally average it and compute AUC
cnn_model_mean_tpr /= n_classes

cnn_model_fpr["macro"] = cnn_model_all_fpr
cnn_model_tpr["macro"] = cnn_model_mean_tpr
cnn_model_roc_auc["macro"] = metrics.auc(cnn_model_fpr["macro"], cnn_model_tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 3
plt.plot(cnn_model_fpr["micro"], cnn_model_tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(cnn_model_roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(cnn_model_fpr["macro"], cnn_model_tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(cnn_model_roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

from itertools import cycle
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(cnn_model_fpr[i], cnn_model_tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, cnn_model_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE MODEL N-Gram CNN, EPOCHS = 100, BATCH SIZE = 30')
plt.legend(loc="lower right")
plt.show()


from keras.callbacks import History
loss = pd.DataFrame(cnn_model.history.history)
loss[['loss', 'val_loss']].plot()

accuracy_cnn_model = pd.DataFrame(cnn_model.history.history)
accuracy_cnn_model[['acc','val_acc']].plot()

loss_acc_cnn_model = pd.DataFrame(cnn_model.history.history)
loss_acc_cnn_model[['acc','loss']].plot()


import matplotlib.pyplot as plt

accu = cnn_model.history['acc']
val_acc = cnn_model.history['val_acc']
loss = model.model['loss']
val_loss = model.model['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#-----new edit code----#
#evalute model preformance
score_cnn_model = cnn_model.evaluate(tweet_text_seq,target,batch_size=None,verbose=2)
print('Test score',score_cnn_model[0])
print('Test accuracy',score_cnn_model[1])

#----------------------
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# filepath="CNN_best_weights.{epoch:02d}-{acc:.4f}.hdf5"
filepath = "CNN_best_weights.hdf5"
# filepath="CNN_best_weights.{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
cnn_model.fit(tweet_text_seq, target, batch_size=32, epochs=25, callbacks=[checkpoint])

# #################################################################vocab creation for Superset X for prediction

sequences_test = tokenizer.texts_to_sequences(tweet_super_set_test)
x_test_seq = pad_sequences(sequences_test, maxlen=110)
loaded_CNN_model = load_model('CNN_best_weights.hdf5')

y_bigram_model_super = loaded_CNN_model.predict(x=x_test_seq)
y_CNN_model_super = model_cnn_01.predict(x=x_test_seq)
y_ptw2_model_super = model_pretrainw2v.predict(x=x_test_seq)


x_super_set_test = tweet_super_set_test[0:23938]
print(len(tweet_super_set_test))
a = x_super_set_test
a = a.to_frame().reset_index(drop=True)

def cat_pred(nda):
    lst=[]
    for i in range(len(nda)):
        #print(nda[i,0], '\t', nda[i,1])
        if nda[i, 0] > nda[i, 1] and nda[i, 0] > nda[i, 2]:
            lst.append(0)
        elif nda[i, 1] > nda[i, 0] and nda[i, 1] > nda[i, 2]:
            lst.append(1)
        elif nda[i, 2] > nda[i, 0] and nda[i, 2] > nda[i, 1]:
            lst.append(2)
    return lst

a['y_bigram_model_result'] = cat_pred(y_bigram_model_super)
a['y_ptw2_model'] = cat_pred(y_ptw2_model_super)
a['CNN_03'] = cat_pred(y_CNN_model_super)

a.to_csv('test_Result.csv', encoding='utf-8')



"""

y_bigram_model_super = loaded_CNN_model.predict(x=x_test_seq)

y_CNN_model_super = model_cnn_03.predict(x=x_test_seq)

y_ptw2_model_super = model_ptw2v.predict(x=x_test_seq)

"""