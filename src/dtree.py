import math
import pickle
import random
import numpy as np
from random import shuffle
from sklearn.metrics import classification_report
from operator import itemgetter


def pickle_operating(fname, item):
    # save or load the pickle file.
    file_name = '%s.pickle' % fname
    print(file_name)
    if not item:
        with open(file_name, 'rb') as fs:
            item = pickle.load(fs)
            return item
    else:
        with open(file_name, 'wb') as fs:
            pickle.dump(item, fs, protocol=pickle.HIGHEST_PROTOCOL)

#the node of the decision tree
class decision_node:
    def __init__(self, feature=-1, val=None, results=None, rb=None, lb=None):
        self.feature = feature
        self.value = val
        self.results = results
        self.leftb = lb
        self.rightb = rb

# splitting the data based on the value of the specific column
def sets_split(data, column, value):
    split_function = lambda row:row[column] >= value
    set1, set2 = [[], []], [[], []]
    for i in range(len(data[0])):
        if split_function(data[0][i]):
            set1[0].append(data[0][i])
            set1[1].append(data[1][i])
        else:
            set2[0].append(data[0][i])
            set2[1].append(data[1][i])
    return [set1, set2]


# calculate the entropy from the data source
def entropy(data, labels):
    results_freq = {}
    ent = 0.0
    for i in range(len(data)):
        if labels[i] in results_freq.keys():
            results_freq[labels[i]] += 1
        else:
            results_freq[labels[i]] = 1

    for label, freq in results_freq.items():
        ent -= float(freq) / len(data) * math.log(float(freq) / len(data), 2)
    return ent


# information gaining from this splitting
def info_gain(set1, set2, data_length, current_score):
    p = float(len(set1)) / data_length
    info = current_score - p * entropy(set1[0], set1[1]) - (1 - p) * entropy(set2[0], set2[1])
    return info


# make decision on the label/category of the data
def to_decide(records, labels):
    res = {}
    for record in records:
        if record in res:
            res[record] += 1
        else:
            res[record] = 1
    sorted_labels = sorted(res.iteritems(), key=itemgetter(1), reverse=True)
    if sorted_labels:
        return sorted_labels[0][0]
    else:
        return random.choice(labels)


def gini_scoring(labels, size, categories):
    score = 0.0
    for cat in categories:
        p = labels.count(cat) / size
        score += p * p
    return score

def gini_index(set1, set2, data_length, categories):
    n_instances = data_length
    gini = 0.0
    #cats = set(labels)
    for sample in [set1, set2]:
        size = float(len(sample[0]))
        if size == 0:
            continue
        #print(len(sample[1]))
        score = gini_scoring(sample[1], size, categories)
        gini += (1.0 - score) * (size / n_instances)
        #print(gini)
    return gini

def splitting_value(column_values, split_decision):
    if split_decision == 'median':
        return np.median(column_values)
    else:
        return np.mean(column_values)

def scoring_methods(data, method, labels, data_length, current_score):
    if method == "entropy":
        score = info_gain(data[0], data[1], data_length, current_score)
    else:
        score = gini_index(data[0], data[1], data_length, labels)
    return score

def score_comparison(method, score1, score2=-1, split=False):
    if split:
        if method == "entropy":
            return score1 > score2
        else:
            return score1 < score2
    else:
        if method == "entropy":
            return score1 > 0
        else:
            return score1 < 1

#splitting the data based on all of the columns
#and find the best splitting
def get_best_split(data, score_method="entropy", split_decision='median'):
    column_count = len(data[0][0])
    labels = set(data[1])
    current_score = entropy(data[0], data[1])
    data_length = len(data[0])
    if score_method=="entropy":
        best = {'score': 0.0, 'feature': None, 'sets': [[[], []], [[], []]]}
    else:
        best = {'score': 2.0, 'feature': None, 'sets': [[[], []], [[], []]]}
    for col in range(0, column_count):
        column_values = [row[col] for row in data[0]]
        value = splitting_value(column_values, split_decision)
        set1, set2 = sets_split(data, col, value)
        info = scoring_methods([set1, set2], score_method, labels, data_length, current_score)
        best_flag = score_comparison(score_method, info, best['score'], True)
        if best_flag and len(set1) > 0 and len(set2) > 0:
            best['score'], best['feature'], best['sets'] = info, (col, value), (set1, set2)
    return best

def grow_tree(data, min_size=10, labels=[], level=0, max_level=5, score_method="entropy", split_method='median'):
    if len(data[0]) < min_size:
        return decision_node(results=to_decide(data[1], labels))
        #return decision_node(results=random.choice(labels))
    if level >= max_level:
        return decision_node(results=to_decide(data[1], labels))
    best = get_best_split(data, score_method, split_method)
    best_flag = score_comparison(score_method, best['score'])
    if best_flag:
        right_branch = grow_tree(best['sets'][0], min_size, labels, level+1, max_level, score_method, split_method)
        left_branch = grow_tree(best['sets'][1], min_size, labels, level+1, max_level, score_method, split_method)
        return decision_node(feature=best['feature'][0], val=best['feature'][1],
                             rb=right_branch, lb=left_branch)
    else:
        return decision_node(results=to_decide(data[1], labels))


# recursive going down the tree to classify the new data point
def predict(node, row):
    if row[node.feature] < node.value:
        if node.leftb.results != None:
            return node.leftb.results
        else:
            return predict(node.leftb, row)
    else:
        if node.rightb.results != None:
            return node.rightb.results
        else:
            return predict(node.rightb, row)

#build the tree on the train data
def experiment(train_data, min_size, depth, score_method, split_method):
    x_train, y_train  = [x[0] for x in train_data], [x[1] for x in train_data]
    labels = list(set(y_train))
    tree = grow_tree([x_train, y_train], min_size, labels, 0, depth, score_method, split_method)
    #pickle_operating('dt_model', tree)
    #print("finished growing the tree!")
    return tree

#predicting the test data based on the tree model
#and evaluated the predictions
def predicting(tree, test_data):
    x_test, y_test = [x[0] for x in test_data], [x[1] for x in test_data]
    predictions = []
    for row in x_test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    #print(classification_report(y_test, predictions))
    return predictions
