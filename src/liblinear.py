import subprocess
from argparse import ArgumentParser
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit
from sklearn.cross_validation import LabelKFold
from utilities import feval, frange, getlabels, writingfile, readfeats


def scaling(output_dir):
    """ Scales features to a specified range (e.g. between -1 and 1).
    Returns 'train.scale' and 'test.scale' scaled feature outputs.
    """
    print 'Scaling features'
    cmd1=["../liblinear/svm-scale", "-l", "-1", "-u", "1", "-s", "../liblinear/range", "../data/"+output_dir+"/training"]
    with open("../data/"+output_dir+"/train.scale", "w") as outfile1:
        subprocess.call(cmd1, stdout=outfile1)
    cmd2=["../liblinear/svm-scale", "-r", "../liblinear/range", "../data/"+output_dir+"/testing"]
    with open("../data/"+output_dir+"/test.scale", "w") as outfile2:
        subprocess.call(cmd2, stdout=outfile2)
    outfile1.close()
    outfile2.close()


def predict(ci,trfile,tfile,pfile):
    """ Fits model and predicts labels for the test data.
    Writes predicted labels to 'pfile', as well as a trained model to 'model'.
    """
    model="../models/"+trfile.split('/')[-1]+'.model'
    traincmd=["../liblinear/train", "-c", "0.001", "-q", trfile, model]
    traincmd[2]=ci
    subprocess.call(traincmd)
    predcmd=["../liblinear/predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    return output


def predict_only(modelfile,tfile,pfile):
    """ Predicts labels for the test data.
    Writes predicted labels to 'pfile'.
    """
    predcmd=["../liblinear/predict", tfile, modelfile, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    return output


def CV(ci,trfile,CV_trfile,CV_tfile,CV_pfile,CV_truey,id_train):
    """ Cross validation over training data, for parameter optimisation.
    Returns accuracy, 3-class f1 and 2-class f1 scores for each iteration.
    """
    feats = readfeats(trfile)
    # ids = readfeats(id_train)
    # cv = LabelKFold(ids, n_folds=5)
    cv = ShuffleSplit(n=len(feats), n_iter=5, test_size=0.4, random_state=0)
    # cv = StratifiedShuffleSplit(y=getlabels(feats), n_iter=5, test_size=0.3, random_state=0)
    acc_list = []
    f1_three_list = []
    f1_two_list = []
    count = 0
    for train_index, test_index in cv:
        count+=1
        cv_trfile = CV_trfile+str(count)
        cv_tfile = CV_tfile+str(count)
        cv_pfile = CV_pfile+str(count)
        cv_truey = CV_truey+str(count)
        X_train = feats[train_index]
        X_test = feats[test_index]
        y_test = getlabels(X_test)
        writingfile(cv_trfile, X_train)
        writingfile(cv_tfile, X_test)
        writingfile(cv_truey, y_test)   
        model="../models/cv/"+cv_trfile.split('/')[-1]+".model"     
        traincmd=["../liblinear/train", "-c", "0.001", "-q", cv_trfile, model]
        traincmd[2]=ci
        subprocess.call(traincmd)
        predcmd=["../liblinear/predict", cv_tfile, model, cv_pfile]
        p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
        output, err = p.communicate()
        y_test, y_predicted = feval(cv_truey, cv_pfile)
        acc_list.append(metrics.accuracy_score(y_test, y_predicted))
        f1_three_list.append(metrics.f1_score(y_test, y_predicted, average='macro'))
        f1_two_list.append((metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2)
    f1_three = np.mean(np.asarray(f1_three_list))
    f1_two = np.mean(np.asarray(f1_two_list))
    acc = np.mean(np.asarray(acc_list))
    print "When C=%s, acc is %f, 2-class-f1 is %f and 3-class-f1 is %f"%(ci, acc, f1_two, f1_three)
    return [acc, f1_three, f1_two] 


def TUNE(trfile,cv_trfile,cv_tfile,cv_pfile,cv_truey,id_train):
    """ Exhaustive grid-search over specified values for C-parameter, for the estimator.
    Returns a list of scores along with its corresponding parameters used, for each cross-validation iteration.
    """
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
    c.sort() #Cost parameter values
    # c=c[:20]
    tunec=[]
    for ci in c:
        tunec.append([ci] + CV(str(ci),trfile,cv_trfile,cv_tfile,cv_pfile,cv_truey,id_train))
    tunec=sorted(tunec,key=lambda x: x[1],reverse=True)
    return tunec


def main(output_dir, ci, steps):
    """ Main function for setting classification process and printing scores.
    """
    print 80*'*'
    print 80*'*'
    trfile = '../data/'+output_dir+'/train.scale'
    tfile = '../data/'+output_dir+'/test.scale'
    pfile = '../data/'+output_dir+'/predresults'
    truefile = '../data/'+output_dir+'/y_test'
    cv_trfile = '../data/'+output_dir+'/cv/train.cv'
    cv_tfile = '../data/'+output_dir+'/cv/test.cv'
    cv_pfile = '../data/'+output_dir+'/cv/predresults.cv'
    cv_truey = '../data/'+output_dir+'/cv/y_test.cv'   
    id_train = '../data/'+output_dir+'/id_train'
    
    if 'scale' in steps:
        print "---Feature scaling"
        scaling(output_dir)
    else:
        pass
        
    if 'tune' in steps:
            print '---Parameter tuning'
            tunec = TUNE(trfile,cv_trfile,cv_tfile,cv_pfile,cv_truey,id_train)
            metric = ['accuracy', '3classf1', '2classf1']
            acc_list = []
            f1_list3 = []
            f1_list2 = []
            for i in range(3):
                pfile = '../data/'+output_dir+'/predresults' + '.' + metric[i]
                tunecc = sorted(tunec,key=lambda x: x[i+1],reverse=True)
                bestc = tunecc[0][0]
                bestCV = tunecc[0][i+1]
                print ""
                print "Five-fold CV on %s, the best %s is %f at c=%f"%(trfile,metric[i],bestCV,bestc)
       
                if 'pred' in steps:
                    print "---Model fitting and prediction"
                    predict(str(bestc),trfile,tfile,pfile)
                    y_test, y_predicted = feval(truefile, pfile)
                    print 'Macro-F1 score: ', metrics.f1_score(y_test, y_predicted, average='macro')
                    print 'Accuracy score: ', metrics.accuracy_score(y_test, y_predicted)
                    print "Macro-F1 score (2 classes):", (metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2
                    print 80*'*'

                    
    if ('tune' not in steps) and ('pred' in steps):
        print "---Model fitting and prediction"
        predict(str(ci),trfile,tfile,pfile)
        y_test, y_predicted = feval(truefile, pfile)

        print "---Evaluation"
        print 'Macro-F1 score: ', metrics.f1_score(y_test, y_predicted, average='macro')
        print 'Accuracy score: ', metrics.accuracy_score(y_test, y_predicted)
        print "Macro-F1 score (2 classes):", (metrics.f1_score(y_test, y_predicted, average=None)[0]+metrics.f1_score(y_test, y_predicted, average=None)[-1])/2
        print 80*'*'
        print 80*'*'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", dest="d", help="Output folder name", default='election')
    parser.add_argument("--c", dest="ci", help="Penalty parameter, C", default=1)
    parser.add_argument("--steps", dest="p", help="Choose classification steps: e.g. scale,tune,pred", default='')
    args = parser.parse_args()
    
    output_dir = args.d + '/output' #All feature outputs and results are stored in this directory
    main(output_dir, args.ci, args.p)

