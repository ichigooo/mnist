from matplotlib import pyplot
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cross_validation
from sklearn.metrics import \
    confusion_matrix,\
    accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %pylab inline

import numpy as np

# organizing data
with open("data/train.csv") as train_data:
    data = np.loadtxt(train_data, delimiter = ',',skiprows = 1)
data.shape
Y_train = data[:,0]

with open("data/test.csv") as test_data:
    X_test = np.loadtxt(test_data, delimiter = ',',skiprows = 1)

# row: 0-(28000-1)
# function: display any mnist digit
def display_mnist(row):
    pyplot.matshow(data[row][1:].reshape(28,28), cmap = "gray")

# find first occurence of each digit
labels = [0,1,2,3,4,5,6,7,8,9]
indices = []
for label in labels:
    for i, digit in enumerate(Y_train):
        if label == digit:
            indices.append(i)
            break
print (indices)

# printing samples of MNIST of each digit
for digit in indices:
    display_mnist(digit)


# function: find the best match
# warmup: 1-NN classifier
# row_num: 0-(28000-1)
def find_best_match(row_num):
    distance = euclidean_distances(X_train, X_train[row_num])
    distance[row_num] = sys.maxsize
    rt = row_num ,Y_train[row_num], np.argmin(distance), Y_train[np.argmin(distance)]
    return rt

# examine prior probability of each digit
digit_class = ["0","1","2","3","4","5","6","7","8","9"]
digit_prob = [0] * 10
for digit in Y_train:
    digit_prob[int(digit)] = digit_prob[int(digit)] + 1
print ("The prior probability of the classes in the training data:")
for i, freq in enumerate(digit_prob):
    print ("for class ", digit_class[i], "the probability is ",freq / 28000)
pyplot.hist(Y_train, normed=1)
pyplot.title("Histogram of Digit Probability")
pyplot.xlabel("Digit")
pyplot.ylabel("Count")
pyplot.xlim(0,9)
pyplot.show()

# 0 and 1 binary test
import itertools

idx_0 = []
idx_1 = []

for i, elem in enumerate(Y_train):
    if elem == 0:
        idx_0.append(i)
    elif elem == 1:
        idx_1.append(i)
zero_data = X_train[idx_0]
one_data = X_train[idx_1]

gen_match = []
distance_0 = []
for i, row in enumerate(zero_data):
    if i < len(zero_data) - 1:
        distance_0.append(euclidean_distances(row, zero_data[i+1:len(zero_data)])[0])
for i, row in enumerate(one_data):
    if i < len(zero_data) - 1:
        distance_0.append(euclidean_distances(row, one_data[i+1:len(one_data)])[0])
list_d = list(distance_0)
gen_match = list(itertools.chain.from_iterable(list_d))
print (gen_match)
distance_1 = []
imp_match = []
for i, row in enumerate(zero_data):
        distance_1.append(euclidean_distances(row, one_data)[0])
list_d = list(distance_1)
imp_match = list(itertools.chain.from_iterable(list_d))

# plot the imposter and genuine matches
bins = np.linspace(0,4500,100)
pyplot.hist(gen_match, bins, alpha = 0.5, normed = 1, label = 'genuine matches')
pyplot.hist(imp_match, bins, alpha = 0.5, normed = 1, label = 'imposter matches')
pyplot.show()

# roc curve
fal_pos_rate = []
tr_pos_rate = []


thr = np.linspace(500, 4000, 100)


for threshold in thr:
    fp = sum(imp_match <= threshold)
    tp = sum(gen_match <= threshold)
    fal_pos_rate.append(fp/len(imp_match))
    tr_pos_rate.append(tp/len(gen_match))

pyplot.scatter(fal_pos_rate, tr_pos_rate)
pyplot.show()


# more code on calculating the error rate, I deleted the code for
# appending to save time when running

# import math
# # roc curve
# # True pos rate = hit rate = TP / P = 1-FNR
# # False pos rate = false acceptance = type I error rate = FP / N =
# # False neg rate = false rejection =type II error rate
# # EER- Equal error rate/ cross over error rate (false pos rate = false neg rate)
# # Accuracy = (TP+TN)/(P+N)
# # tried several intervals, and finally got the error rate
#
# thr = np.linspace(2600, 2650, 1000)
#
# eer = 0
# for threshold in thr:
#     fp = sum(imp_match <= threshold)
#     tp = sum(gen_match <= threshold)
#     tn = sum(imp_match > threshold)
#     fn = sum(gen_match > threshold)
#     fpr = fp / float(fp + tn)
#     fnr = fn / float(fn + tp)
#     if abs(fpr - fnr) < 0.01:
#         eer = fpr
# print (eer)


# implement kNN classifier:
def get_k_neighbors(train, test_row, k):
    distance = euclidean_distances(train, test_row)
    idx_dis = np.argpartition(distance.T, k)
    return (idx_dis.T[0:k]).T

from collections import Counter

def get_response(k_dis, label):
    k_target = label[k_dis]
    result = Counter(k_target[0]).most_common()
    return result[0][0]


# do 3-fold validation in the training data
all_folds = cross_validation.KFold(len(X_train), n_folds=3)

Y_pred = []
Y_true = []

for train, test in all_folds:
    for index in test:
        Y_pred.append(get_response(get_k_neighbors(X_train[train], X_train[index], 1), Y_train[train]))
        Y_true.append(Y_train[index])
confusion_matrix(Y_true, Y_pred)
accuracy_score(Y_true, Y_pred)



# using kNN to get the labels for X_test
Y_test = []
for row in X_test:
    Y_test.append(get_response(get_k_neighbors(X_train, row, 1), Y_train))
print (len(Y_test))
id = [i for i in range(1,len(Y_test)+1)]
Y_test = np.insert(Y_test, 0, id, axis=1)
np.savetxt("Result1.1(2).csv",Y_test,fmt='%i',header='ImageId,Label', delimiter=',')
# the results are in Y_test
