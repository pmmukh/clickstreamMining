import random
import pickle as pkl
import argparse
import csv
import numpy as np
import math
import copy
import scipy
from scipy.stats import chisquare

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''


# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)


# loads Train and Test data
def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest = [], [], []
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtrain.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtest.append(rw)

    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytrain.append(rw)

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest


num_feats = 274


def test(testrow, node):
    if node.data == "T":
        return 1
    elif node.data == "F":
        return 0
    val = testrow[node.data]
    # print(str(node.data))
    if not node.nodes:
        return None
    if node.nodes[val - 1] == None or node.nodes[val - 1] == -1:
        return None
    else:
        return test(testrow, node.nodes[val - 1])



def calc_domain(f,rowset=[]):
    domain=[]
    i=0
    while i<len(rowset):
        if rowset[i][f] not in domain:
            domain.append(rowset[i][f])
            #if len(domain)==5:
                #return domain
        i=i+1
    return domain


def calc_entropy(rowset=[]):
    p=rowset.count(1)
    n=rowset.count(0)

    #print p,n
    if p==0 or n==0 :
        return 0

    #float(p_prob)

    #float(n_prob)
    p_prob=float(p)/float(p+n)
    n_prob=float(n)/float(p+n)

    #print p_prob,n_prob

    #print math.log(p/(p+n),2),math.log(n/(p+n),2)
    entropy=(-1 * p_prob*math.log(p_prob,2)) + (-1 * n_prob*math.log(n_prob,2))

    return entropy


def calc_probability(f,fval,rowset=[]):
    i=0
    ctr=0
    while i < len(rowset):
        if rowset[i][f]==fval:
            ctr=ctr+1
        i=i+1

    N=float(len(rowset))
    probability=float(float(ctr)/float(N))
    return probability

def calc_conditional_entropy(f,rowset=[],label=[]):
    #domain=calc_domain(f,rowset)
    conditional_entropy=0
    i=1
    while i<=5:
        probability=calc_probability(f,i,rowset)
        templabel=[]
        j=0
        while j < len(rowset):
            if rowset[j][f]==i :
                templabel.append(label[j])
            j=j+1
        if len(templabel) > 0 :
            entropy=calc_entropy(templabel)
            conditional_entropy=conditional_entropy+(probability * entropy)
        i=i+1
    return conditional_entropy


def calc_chisquare_usinglib(f,features=[],label=[]):
    i=1
    exp_array=[]
    actual_array=[]
    if label.count(0) == 0 or label.count(1)==0 :
        return 0
    while i <= 5:
        newfeatures=[]
        newlabel=[]
        j=0
        while j < len(features):
            if features[j][f] == i:
                newfeatures.append(features[j])
                newlabel.append(label[j])
            j=j+1
        if len(newlabel) > 0:
            expectedpos=float(label.count(1)) * float((len(newlabel))/float(len(label)))
            expectedneg=float(label.count(0)) * float((len(newlabel))/float(len(label)))
            exp_array.append(expectedpos)
            exp_array.append(expectedneg)
            actual_array.append(newlabel.count(1))
            actual_array.append(newlabel.count(0))
        i=i+1

    s,p=chisquare(actual_array,exp_array)
    return p


def calc_chisquare(f,features=[],label=[]):
    i=1
    #exp_array=[]
    #actual_array=[]
    s=0
    if label.count(0) == 0 or label.count(1)==0 :
        return 0
    while i <= 5:
        newfeatures=[]
        newlabel=[]
        j=0
        while j < len(features):
            if features[j][f] == i:
                newfeatures.append(features[j])
                newlabel.append(label[j])
            j=j+1
        if len(newlabel) > 0:
            expectedpos=float(label.count(1)) * float((len(newlabel))/float(len(label)))
            expectedneg=float(label.count(0)) * float((len(newlabel))/float(len(label)))

            pos=float(newlabel.count(1))
            neg=float(newlabel.count(0))
            chisquare= float((float(expectedpos - pos) * float(expectedpos - pos))/expectedpos) + float((float(expectedneg - neg) * float(expectedneg - neg))/expectedneg)
            s=float(s+chisquare)



            #exp_array.append(expectedpos)
            #exp_array.append(expectedneg)
            #actual_array.append(newlabel.count(1))
            #actual_array.append(newlabel.count(0))
        i=i+1

    #s,p=chisquare(actual_array,exp_array)
    return s


def calc_chisquare_prob(val,chi_square={}):
    ctr=0
    all_item=[]
    s=chi_square[val]
    for i in chi_square.keys():
        all_item.append(i)
        if chi_square[i] >= s:
            ctr= ctr + 1
    prob=float(float(ctr)/float(len(all_item)))
    return prob

def get_splitnode(processed,features=[],label=[]):
    split_node = -1
    f = 0
    feature_count = len(features[0])
    gain = 0
    while f < feature_count:
        if f in processed:
            f = f + 1
            continue
        conditional_entropy = calc_conditional_entropy(f, features, label)
        entropy = calc_entropy(label)
        temp_gain = entropy - conditional_entropy
        if temp_gain >= gain:
            gain = temp_gain
            split_node = f

        f = f + 1

    return split_node



def populate_tree(threshold,processed=[],features=[],label=[]):
    split_node=get_splitnode(processed,features,label)

    processed.append(split_node)
    d_node=TreeNode(data=split_node)

    child_features=[]
    child_labels=[]

    chi_square={}
    chi_prob={}
    i=1
    while i<=5:
        newlabel=[]
        newfeature=[]
        row=0
        while row < len(features):
            if features[row][split_node]==i:
                newfeature.append(features[row])
                newlabel.append(label[row])
            row=row+1
        child_features.append(newfeature)
        child_labels.append(newlabel)
        if len(newlabel) > 0:
            next_split=get_splitnode(processed,newfeature,newlabel)
            #chi_prob[i]=calc_chisquare_usinglib(next_split,newfeature,newlabel)
            chi_square[i] = calc_chisquare(next_split, newfeature, newlabel)
        i=i+1

    for i in chi_square.keys():
        chi_prob[i]=calc_chisquare_prob(i,chi_square)

    i=1
    while i <= 5 :
        #print "\nsplitnode=",split_node,"\nnew feature count=",len(newfeature)
        #print "\nnew label:\n",newlabel

        if i not in chi_prob.keys():   #
            i=i+1
            continue

        if chi_prob[i] > threshold:
            if child_labels[i-1].count(0) >= child_labels[i-1].count(1) :
                d_node.nodes[i-1]=TreeNode("F",[])
            else :
                d_node.nodes[i - 1] = TreeNode("T",[])
            i=i+1
            continue

        entropy=calc_entropy(child_features[i - 1])
        #print "\nentropy:",entropy
        if entropy > 0:
            temp=[]
            for proc in processed:
              temp.append(proc)
            node = populate_tree(threshold,temp,child_features[i-1],child_labels[i-1])
            d_node.nodes[i-1] = node
            #if child_ctr < 4 :
                #child_ctr=child_ctr+1
        else :
            #val="T"
            if child_labels[i - 1].count(0) < child_labels[i - 1].count(1) :
                #val="T"
                node = TreeNode("T", [])
                d_node.nodes[i-1] = node
                #if child_ctr < 4:
                    #child_ctr = child_ctr + 1
            else:
                #val="F"
                node=TreeNode("F",[])
                d_node.nodes[i-1]=node
                #if child_ctr < 4:
                    #child_ctr=child_ctr+1
        i=i+1

    return d_node


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)
threshold=pval

print "Threshold: ",threshold,"\n"

tn = populate_tree(threshold,[],Xtrain,Ytrain)
tn.save_tree("tree.pkl")
print("Testing...")
# generate random labels
Ypredict = []
for i in range(0,len(Xtest)) :
    Ypredict.append([test(Xtest[i], tn)])
# print(Ypredict)
print(len(Ypredict))
Ytest_predict_name = "output.csv"
with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")



