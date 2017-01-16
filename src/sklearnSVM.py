import os, time
from argparse import ArgumentParser
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, GroupKFold
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from utilities import readfeats, readfeats_sklearn, twoclass_fscore, frange
# from liblinear import scaling



def macro_averaged_precision(y_true, y_predicted):
    p = metrics.precision_score(y_true, y_predicted, average='macro')
    return p


def predict(clf, x_train, y_train, x_test, y_test):
    y_predicted = clf.predict(x_test)
    print 'Macro-F1 score: ', metrics.f1_score(y_test, y_predicted, average='macro')
    print 'Accuracy score: ', metrics.accuracy_score(y_test, y_predicted)
    print "Macro-F1 score (2 classes):", (metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2


def CV(x_train, y_train):
    c=[]
    crange=frange(0.00001,1,10)
    c.extend([i for i in crange])
    crange=frange(0.00003,3,10)
    c.extend([i for i in crange])
    crange=frange(0.00005,5,10)
    c.extend([i for i in crange])
    crange=frange(0.00007,7,10)
    c.extend([i for i in crange])
    crange=frange(0.00009,10,10)
    c.extend([i for i in crange])
    c.sort()
    # c = c[:20]
    
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0).split(x_train, y_train)
    # ids = readfeats('../data/election/output/id_train') # only for election data
    # cv = GroupKFold(n_splits=5).split(x_train, y_train, ids)
    clf = svm.LinearSVC()
    param_grid = [{'C': c, 'class_weight': ['balanced']}]

    twoclass_f1_macro = metrics.make_scorer(twoclass_fscore, greater_is_better=True)
    precision_macro = metrics.make_scorer(macro_averaged_precision, greater_is_better=True)
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv, verbose=0, scoring='accuracy')
    
    grid_search.fit(x_train, y_train)
    print("Best parameters set:")
    print '\n'
    print(grid_search.best_estimator_)
    print '\n'
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    print '\n'
    return grid_search.best_estimator_


def save_model(clf, filepath):
    joblib.dump(clf, filepath)


def main(output_dir):
    trfile = '../data/'+output_dir+'/train.scale'
    tfile = '../data/'+output_dir+'/test.scale'
    pfile = '../data/'+output_dir+'/predresults'
    truefile = '../data/'+output_dir+'/y_test'  

    # print "scaling features"
    # scaling(output_dir)

    print "extracting features for training"
    x_train, y_train = readfeats_sklearn(trfile)
    print "extracting features for testing"
    x_test, y_test = readfeats_sklearn(tfile)
    
    print "cross-validation"
    clf = CV(x_train, y_train)
    # print "saving trained model"
    # save_model(clf, '')
    print "evaluation"
    predict(clf, x_train, y_train, x_test, y_test)
    

if __name__ == "__main__":
    start = time.clock()
    parser = ArgumentParser()
    parser.add_argument("--data", dest="d", help="Output folder name", default='election')
    args = parser.parse_args()
    output_dir = args.d + '/output'
    main(output_dir)
    
    print "\n"
    print "Time taken:", time.clock() - start