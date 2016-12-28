from sklearn import metrics
from sklearn.datasets import load_svmlight_file
import networkx as nx
from networkx.algorithms import traversal
import numpy as np
import os
import codecs
import collections


def twoclass_fscore(y_predicted, y_true):
    f = (metrics.f1_score(y_true, y_predicted, average=None)[0]+metrics.f1_score(y_true, y_predicted, average=None)[-1])/2
    return f


def feval(truefile, predfile):
    truefile = os.path.abspath(truefile)
    predfile = os.path.abspath(predfile)
    f1 = open(truefile, 'r')
    f2 = open(predfile, 'r')
    l1 = f1.readlines()
    l2 = f2.readlines()
    y_test = []
    y_predicted = []
    if len(l1) == len(l2):
        for i in xrange(len(l1)):
            y_test.append(int(l1[i].strip()))
            y_predicted.append(int(l2[i].strip()))
    else:
        raise Exception('ERROR: true and pred file length do not match!')
    f1.close()
    f2.close()
    return y_test, y_predicted


def getlabels(X):
    """ Extracts labels
    """
    y = []
    for i in X:
        i = i[0].split(' ')
        y.append(int(i[0]))
    return y


def writingfile(filepath, X):
    """Writes file
    """
    with open(filepath,'w') as f:
        for item in X:
            f.write("%s\n" % item)


def readfeats(filepath):
    """ Reads feature file
    """
    f = open(filepath, 'r')
    lines = f.readlines()
    d = []
    for i in lines:
        d.append(i.strip())
    d = np.asarray(d)
    return d


def readfeats_sklearn(datapath):
    """ Reads LibSVM format file into numpy array 
        feature vectors and labels
    """
    data = load_svmlight_file(datapath)
    return data[0].toarray(), data[1]


def readConll(filepath, pmode):
    """ Reads parse conll file
    """
    conll=[]
    with open(filepath, 'r') as fin:
        buf = []
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                if len(buf) > 0:
                    conll.append(buf)
                    buf = []
                continue
            line = line.split()
            if pmode == 'stanford':
                position, token, _, tag, _, _, parser, rel, _, _ = line # for stanford parses
            elif pmode == 'cmu':
                position, token, _, tag, _, _, parser, rel = line # for cmu parses
            position = int(position)
            parser = int(parser)
            buf.append((position, token.lower(), tag, parser, rel))
    return conll


def frange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r *= step  


def writevec(filename,x,y):
    """Writes feature matrix in LibSVM format
    """
    f=open(filename,'wb')
    for i in xrange(len(y)):
            f.write(str(y[i])+'\t')
            feature=x[i]
            for (j,k) in enumerate(feature):
                    f.write(str(j+1)+':'+str(k)+' ')
            f.write('\n')
    f.close() 


def writey(filename, y):
    """Writes labels
    """
    f=open(filename,'wb')
    for i in y:
            f.write(str(i)+'\n')
    f.close()     
        
        

def concat_vec(x1, f2):
    """Concatenates numpy vectors
    """
    x2 = np.load(f2)
    x = np.concatenate((x1, x2), axis=1)
    return x



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)   


def traversaltree(conll,target,empty,pmode):
    """Processes conll-format parse data and outputs words with corresponding POS tags
    as well as the index position of the target entity
    """
    G = nx.Graph()
    for position, token, tag, parser, rel in conll:
        G.add_node(position)
        for position1, token1, tag1, parser1, rel1 in conll:
            if position1 == parser:
               head = position1
        if (parser == 0) or (parser == -1):
           pass
        else:
           try:
               G.add_edge(position, head, label=rel)
           except:
               print token
               print conll
    target_positions = []
    for position, token, tag, parser, rel in conll:
      if token == target:
        target_positions.append(position)

    if pmode=='cmu':
        positions = [[item for sublist in traversal.bfs_successors(G, target_position).values() for item in sublist] for target_position in target_positions]
    elif pmode=='stanford':
        positions = [[item for sublist in [traversal.bfs_successors(G, target_position).keys()] for item in sublist] for target_position in target_positions]

    words = [[conll[i-1][1] for i in position] for position in positions]
    words_tags = [[(conll[i-1][1], conll[i-1][2]) for i in position] for position in positions]

    if not words[0]:
      empty = True
    result = []
    new_target_positions = []
    for i in range(len(words_tags)):
       temp={}
       for j in range(len(words_tags[i])):
         temp.update({positions[i][j]: (words_tags[i][j][0], words_tags[i][j][1])})
       temp.update({target_positions[i]: (target, conll[target_positions[i]-1][2])})
       temp = collections.OrderedDict(sorted(temp.items())).values()
       result.append(temp)
       new_target_positions.append(temp.index((target, conll[target_positions[i]-1][2])))

    return result, new_target_positions, empty