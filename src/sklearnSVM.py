import os, time
from argparse import ArgumentParser
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, GroupKFold
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from utilities import readfeats, readfeats_sklearn, twoclass_fscore, frange, writingfile
# from liblinear import scaling



def macro_averaged_precision(y_true, y_predicted):
    p = metrics.precision_score(y_true, y_predicted, average='macro')
    return p


def predict(clf, x_train, y_train, x_test, y_test):
    y_predicted = clf.predict(x_test)
    print 'Macro-F1 score: ', metrics.f1_score(y_test, y_predicted, average='macro')
    print 'Accuracy score: ', metrics.accuracy_score(y_test, y_predicted)
    print "Macro-F1 score (2 classes):", (metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2
    return y_predicted


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
    c.sort() #Cost parameter values; use a bigger search space for better performance
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(x_train, y_train)
    # ids = readfeats('../data/election/output/id_train') # only for election data
    # cv = GroupKFold(n_splits=5).split(x_train, y_train, ids)
    clf = svm.LinearSVC()
    param_grid = [{'C': c}]

    twoclass_f1_macro = metrics.make_scorer(twoclass_fscore, greater_is_better=True)
    precision_macro = metrics.make_scorer(macro_averaged_precision, greater_is_better=True)
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv, verbose=0, scoring='f1_macro')
    
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

    print "loading features for training"
    x_train, y_train = readfeats_sklearn(trfile)
    print "loading features for testing"
    x_test, y_test = readfeats_sklearn(tfile)
    
    print "cross-validation"
    clf = CV(x_train, y_train) # Comment this if parameter tuning is not desired

    # print "training classifier"
    # clf = svm.LinearSVC(C=1, class_weight='balanced') # Manually select C-parameter for training SVM
    # clf.fit(x_train, y_train)

    # print "saving trained model"
    # save_model(clf, '../models/sklearn_saved.model')

    print "evaluation"
    preds = predict(clf, x_train, y_train, x_test, y_test)

    print "writing labels"
    writingfile(pfile, preds)
    

if __name__ == "__main__":
    start = time.clock()
    parser = ArgumentParser()
    parser.add_argument("--data", dest="d", help="Output folder name", default='election')
    args = parser.parse_args()
    output_dir = args.d + '/output'
    main(output_dir)
    
    print "\n"
    print "Time taken:", time.clock() - start