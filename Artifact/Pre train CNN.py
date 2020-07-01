import os
path_ = 'C:/Users/User/Downloads'
os.chdir(path_)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# ****************************** READING FILE **************
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



# ******************************************************************************

x_super_set_test = main_df.Tweet_Text  # to be used for prediction
x = main_df.Tweet_Text  # to be used for Training
y = main_df.target  # to be used for training

# ***********************

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils


def labelize_tweets_ug(tweets, label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


cores = multiprocessing.cpu_count()

# ********************** Labelizing, Vocab creation for X ************
all_x_w2v = labelize_tweets_ug(x, 'all')
# _____________________________ug_cbow ______________________________
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2,
                         min_count=2, workers=cores, alpha=0.065,
                         min_alpha=0.065)  # x values
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])  # vocab creation x
for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),
                        total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha
### ___________________ug_sg __________________
model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2,
                       workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),
                      total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

# ------------- Saving the Word to Vector model ------------------
'''****************************************
Word to Vector converts each word to a unique integer value in wv.vocab
and vectors are contained in wv vectors
*****************************************'''
model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
model_ug_sg.save('w2v_model_ug_sg.word2vec')
# ----------------------------------------------------------------

# ------------- Loading Word-Vector Model ------------------------
from gensim.models import KeyedVectors

model_ug_cbow = KeyedVectors.load(r'w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load(r'w2v_model_ug_sg.word2vec')
# ----------------------------------------------------------------

# -------------- Embedding index creation for Training X -------------------
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w], model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))
# --------------------------------------------------------------------------

# ________________________________ CNN KERAS NETWORK for ________________
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=90000)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)  # tokenize traing  x

length = []
for x_ in x:
    length.append(len(x_.split()))
max(length)

x_seq = pad_sequences(sequences, maxlen=200)  # -- x_seq-------- taininf=g data
print('Shape of data tensor:', x_seq.shape)

num_words = 90000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
np.array_equal(embedding_matrix[6], embeddings_index.get('climate'))

seed = 7

# X_train_seq = x_seq

# --------------------------- KERAS NETWORK ---------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

model_pretrainw2v = Sequential()
e = Embedding(90000, 200, weights=[embedding_matrix], input_length=200, trainable=True)
model_pretrainw2v.add(e)
model_pretrainw2v.add(Flatten())
model_pretrainw2v.add(Dense(100, activation='relu'))
model_pretrainw2v.add(Dense(50, activation='relu'))
model_pretrainw2v.add(Dense(3, activation='softmax'))
model_pretrainw2v.summary()
model_pretrainw2v.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_pretrainw2v.fit(x_seq, y, epochs=30, batch_size=32, verbose=2)

structure_test = Sequential()
e = Embedding(90000, 200, input_length=200)
structure_test.add(e)
structure_test.add(Conv1D(filters=50, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.add(GlobalMaxPooling1D())
structure_test.summary()
#---------------------------------------evaluation-----------------------#

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report

#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(x_seq, y, test_size= 0.3, random_state = 24)

# ************---------- fit model CNN03---------------------*********************
batch_size = 20
model_pretrainw2v.fit(X_train, Y_train, epochs=100, verbose=1, batch_size=batch_size,validation_data= (X_test, Y_test),callbacks=[early_stop])
# ************---------------------------------------------------------------------
# ------------------------------- ONE HOT ENCODING ------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(Y_test)
model_pretrainw2v_y_test = encoder.transform(Y_test)
model_pretrainw2v_y_test = np_utils.to_categorical(model_pretrainw2v_y_test)
# ____________________________ PREDICTING O TEST TEST _____________________________

#analyze the results
model_pretrainw2v_y_pred = model_pretrainw2v.predict(X_test)
model_pretrainw2v_y_prob = model_pretrainw2v.predict_proba(X_test)

model_pretrainw2v_score, model_pretrainw2v_acc = model_pretrainw2v.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("score", model_pretrainw2v_score)
print("Accuracy", model_pretrainw2v_acc)
# ____________________________Classification report_______________________________________________________

print(classification_report(Y_test, np.argmax(model_pretrainw2v_y_pred, axis = 1)))
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


model_pretrainw2v_y_prob_F1 = cat_pred(model_pretrainw2v_y_prob)
model_pretrainw2v_y_pred_F1 = cat_pred(model_pretrainw2v_y_pred)
model_pretrainw2v_f1_score = f1_score(Y_test, model_pretrainw2v_y_prob_F1, average='macro')
model_pretrainw2v_f1_micro_ovo = f1_score(Y_test, model_pretrainw2v_y_pred_F1, average='micro')

#-----------------------Im Able to print this one ------------------------------------------------------------#

macro_roc_auc_ovo = roc_auc_score(model_pretrainw2v_y_test, model_pretrainw2v_y_prob, multi_class="ovo", average="macro")
macro_roc_auc_ovr = roc_auc_score(model_pretrainw2v_y_test, model_pretrainw2v_y_pred, multi_class="ovr", average="macro")
weighted_roc_auc_ovo = roc_auc_score(model_pretrainw2v_y_test, model_pretrainw2v_y_prob, multi_class="ovo", average="weighted")
weighted_roc_auc_ovr = roc_auc_score(model_pretrainw2v_y_test, model_pretrainw2v_y_pred, multi_class="ovr", average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

# ----------------- ROC curve -------------------------------------------
# Compute ROC curve and ROC area for each class
model_pretrainw2v_fpr = dict()
model_pretrainw2v_tpr = dict()
model_pretrainw2v_roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    model_pretrainw2v_fpr[i], model_pretrainw2v_tpr[i], _ = roc_curve(model_pretrainw2v_y_test[:, i], model_pretrainw2v_y_pred[:, i])
    model_pretrainw2v_roc_auc[i] = metrics.auc(model_pretrainw2v_fpr[i], model_pretrainw2v_tpr[i])

# Compute micro-average ROC curve and ROC area
model_pretrainw2v_fpr["micro"], model_pretrainw2v_tpr["micro"], _ = roc_curve(model_pretrainw2v_y_test.ravel(), model_pretrainw2v_y_pred.ravel())
model_pretrainw2v_roc_auc["micro"] = metrics.auc(model_pretrainw2v_fpr["micro"], model_pretrainw2v_tpr["micro"])


# First aggregate all false positive rates
model_pretrainw2v_all_fpr = np.unique(np.concatenate([model_pretrainw2v_fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
model_pretrainw2v_mean_tpr = np.zeros_like(model_pretrainw2v_all_fpr)
for i in range(n_classes):
    model_pretrainw2v_mean_tpr += interp(model_pretrainw2v_all_fpr, model_pretrainw2v_fpr[i], model_pretrainw2v_tpr[i])

# Finally average it and compute AUC
model_pretrainw2v_mean_tpr /= n_classes

model_pretrainw2v_fpr["macro"] = model_pretrainw2v_all_fpr
model_pretrainw2v_tpr["macro"] = model_pretrainw2v_mean_tpr
model_pretrainw2v_roc_auc["macro"] = metrics.auc(model_pretrainw2v_fpr["macro"], model_pretrainw2v_tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 3
plt.plot(model_pretrainw2v_fpr["micro"], model_pretrainw2v_tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(model_pretrainw2v_roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(model_pretrainw2v_fpr["macro"], model_pretrainw2v_tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(model_pretrainw2v_roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

from itertools import cycle
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(model_pretrainw2v_fpr[i], model_pretrainw2v_tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, model_pretrainw2v_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE MODEL Pre-train word2vec, EPOCHS = 100')
plt.legend(loc="lower right")
plt.show()

