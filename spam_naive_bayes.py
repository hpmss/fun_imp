import numpy as np
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score

path = "D:\\hpms\\mlearn\\ex6\\"
train_data_fn = 'train-features.txt'
test_data_fn = 'test-features.txt'
train_label_fn = 'train-labels.txt'
test_label_fn = 'test-labels.txt'

N = 2500

def read_data(data_fn,label_fn):
    with open(path + label_fn) as fh:
        content = fh.readlines()
    label = [int(x.strip()) for x in content]

    with open(path + data_fn) as fh:
        content = fh.readlines()
    content = [x.strip() for x in content]

    data = np.zeros((len(content),3),dtype = int)

    for i,line in enumerate(content):
        a = line.split(' ')
        data[i, :] = np.array([int(a[0]),int(a[1]),int(a[2])])

    data = coo_matrix((data[:, 2], (data[:, 0] - 1, data[:, 1] - 1)),\
             shape=(len(label), N))
    return (data, label)

train_data,train_label = read_data(train_data_fn,train_label_fn)
test_data, test_label  = read_data(test_data_fn, test_label_fn)

clf = MultinomialNB()
clf.fit(train_data,train_label)

y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % \
      (train_data.shape[0],accuracy_score(test_label, y_pred)*100))
