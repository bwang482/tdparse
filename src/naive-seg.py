import sys
sys.path.insert(0, r'../')
import os
import re
import numpy as np
import gensim
from collections import defaultdict
import argparse
from argparse import ArgumentParser
from utilities import readConll, writevec, writey, traversaltree
from data.dataprocessing import streamtw, streamtwElec
from tdparse import lexicon, streamdata, readTang
import warnings

warnings.simplefilter("error")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def segment(tw, target):
    tw = ' '.join(tw)
    segs = []
    for seg in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\,)\s', tw):
        for token in seg.split():
            if token == target:
              segs.append(seg.split())
              break
    target_positions = [[i for i, j in enumerate(seg) if j == target] for seg in segs]

    new_segs = []
    for i in xrange(len(target_positions)):
      temp = []
      if len(target_positions[i]) > 1:
        for j in xrange(len(target_positions[i])):
          temp.append(segs[i])
      else:
        temp.append(segs[i])
      new_segs.append(temp)
    
    target_positions = [item for sublist in target_positions for item in sublist]
    new_segs = [item for sublist in new_segs for item in sublist]

    return new_segs, target_positions

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class targettw(object):
    def __init__(self,w2vf='../resources/wordemb/w2v/c10_w3_s100',
                        sswef='../resources/wordemb/sswe'):
        self.w2v=gensim.models.Word2Vec.load(w2vf) 
        self.sswe=readTang(sswef) 
        self.lexicons=lexicon()

    def emdsswe(self,i,loc,uni,target):
        f=np.array([])    
        l=np.array([])
        r=np.array([])
        t=np.array([])
        ls=np.array([])
        rs=np.array([])
        #embeddings of fulltw features
        f=self.sswe.get(uni,self.sswe['<unk>'])
        #embeddings of  left context features
        if i<loc:
            l=self.sswe.get(uni,self.sswe['<unk>'])
            if self.lexicons.pos.issuperset(set([uni])):
                try:
                    ls=self.sswe[uni]
                except:
                    pass
            if self.lexicons.neg.issuperset(set([uni])):
                try:
                    ls=self.sswe[uni]
                except:
                    pass
        #embbeddings of  target features  
        elif(i==loc):
            t=self.sswe.get(target.replace('_',''),self.sswe['<unk>'])               
            target2=target.split('_')
            for wd in target2:
                ti=self.sswe.get(wd,self.sswe['<unk>'])
                t=np.concatenate([t,ti])
            
        #embbeddings of  right context features
        else:
            r=self.sswe.get(uni,self.sswe['<unk>'])
            if self.lexicons.pos.issuperset(set([uni])):
                try:
                    rs=self.sswe[uni]
                except:
                    pass
            if self.lexicons.neg.issuperset(set([uni])):
                try:
                    rs=self.sswe[uni]
                except:
                    pass 
        return [f,l,t,r,ls,rs]
    def emdw2v(self,i,loc,uni,target):
        f=np.array([])    
        l=np.array([])
        r=np.array([])
        t=np.array([])
        ls=np.array([])
        rs=np.array([])
        try:
            f=self.w2v[uni]
        except:
            pass 
        #embbeddings of  left context features
        if i<loc:
            try:
                l=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.pos.issuperset(set([uni])):
                    ls=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.neg.issuperset(set([uni])):
                    ls=self.w2v[uni]
            except:
                pass
        #embbeddings of  target feature  
        elif(i==loc):
            try:
                t=self.w2v[target.replace('_','')]
            except:
                pass                
            target2=target.split('_')
            for wd in target2:
                try:
                    ti=self.w2v[wd]
                    t=np.concatenate([t,ti])
                except:
                    pass    
        #embbeddings of  right context features
        else:
            try:
                r=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.pos.issuperset(set([uni])):
                    rs=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.neg.issuperset(set([uni])):
                    rs=self.w2v[uni]
            except:
                pass   
        return [f,l,t,r,ls,rs]        

    def concattw(self,feature,size,tws,etype,locs,target,emp,emode):
        """ Concatenation of different features
        """
        fullf=np.array([])
        leftf=np.array([])
        rightf=np.array([])
        tarf=np.array([])
        fulltw_maxs=np.array([])
        fulltw_mins=np.array([])
        fulltw_means=np.array([])
        fulltw_prods=np.array([])
        left_maxs=np.array([])
        left_mins=np.array([])
        left_means=np.array([])
        left_prods=np.array([])
        right_maxs=np.array([])
        right_mins=np.array([])
        right_means=np.array([])
        right_prods=np.array([])
        tar_maxs=np.array([])
        tar_mins=np.array([])
        tar_means=np.array([])
        tar_prods=np.array([])
        leftsenti_max=np.array([])
        rightsenti_max=np.array([])
        leftsenti_sum=np.array([])
        rightsenti_sum=np.array([])
        fulltws=np.array([])
        lefts=np.array([])
        rights=np.array([])
        tars=np.array([])
        leftsentis=np.array([])
        rightsentis=np.array([])
        for a in range(len(locs)):
          if emode=='full':
            tw = tws
          else:
            tw = tws[a]
          loc = locs[a]
          fulltw=np.array([])
          left=np.array([])
          right=np.array([])
          tar=np.array([])
          leftsenti=np.array([])
          rightsenti=np.array([])
          for i,uni in enumerate(tw):
              if etype=='w2v':
                  f,l,t,r,ls,rs=self.emdw2v(i,loc,uni,target)
              if etype=='sswe':
                  f,l,t,r,ls,rs=self.emdsswe(i,loc,uni,target)
              fulltw=np.concatenate([fulltw,f])
              left=np.concatenate([left,l])
              tar=np.concatenate([tar,t])
              right=np.concatenate([right,r])
              leftsenti=np.concatenate([leftsenti,ls])
              rightsenti=np.concatenate([rightsenti,rs])
          #padding 
          if list(left)==[]:
              left=np.zeros((2*size,))
          if list(right)==[]:
              right=np.zeros((2*size,))
          if list(fulltw)==[]:
              fulltw=np.zeros((2*size,)) 
          if list(tar)==[]:
              tar=np.zeros((2*size,))
          
          if len(left)<=size:
              left=np.concatenate([left,np.zeros((size,))])
          if len(right)<=size:
              right=np.concatenate([right,np.zeros((size,))])
          if len(fulltw)<=size:
              fulltw=np.concatenate([fulltw,np.zeros((size,))])
          if len(tar)<=size:
              tar=np.concatenate([tar,np.zeros((size,))])   

          if list(leftsenti)==[]:
              leftsenti=np.zeros((size,))
          if list(rightsenti)==[]:
              rightsenti=np.zeros((size,))       

          fullf=np.append(fullf,fulltw, axis=0)
          leftf=np.append(leftf,left, axis=0)
          rightf=np.append(rightf,right, axis=0)
          tarf=np.append(tarf,tar, axis=0)
          fulltw=fulltw.reshape(len(fulltw)/size,size)
          left=left.reshape(len(left)/size,size)
          right=right.reshape(len(right)/size,size) 
          tar=tar.reshape(len(tar)/size,size)
          leftsenti=leftsenti.reshape(len(leftsenti)/size,size)   
          rightsenti=rightsenti.reshape(len(rightsenti)/size,size)
          
          fulltw_maxs=np.append(fulltw_maxs,fulltw.max(axis=0), axis=0)
          fulltw_mins=np.append(fulltw_mins,fulltw.min(axis=0), axis=0)
          fulltw_means=np.append(fulltw_means,fulltw.mean(axis=0), axis=0)
          fulltw_prods=np.append(fulltw_prods,fulltw.prod(axis=0), axis=0)
          
          left_maxs=np.append(left_maxs,left.max(axis=0), axis=0)
          left_mins=np.append(left_mins,left.min(axis=0), axis=0)
          left_means=np.append(left_means,left.mean(axis=0), axis=0)
          if not np.count_nonzero(left):
            left_prods=np.append(left_prods,left.prod(axis=0), axis=0)
          else:
            left_prods=np.append(left_prods,left[~np.all(left == 0, axis=1)].prod(axis=0), axis=0)
          right_maxs=np.append(right_maxs,right.max(axis=0), axis=0)
          right_mins=np.append(right_mins,right.min(axis=0), axis=0)
          right_means=np.append(right_means,right.mean(axis=0), axis=0)
          if not np.count_nonzero(right):
            right_prods=np.append(right_prods,right.prod(axis=0), axis=0)
          else:
            right_prods=np.append(right_prods,right[~np.all(right == 0, axis=1)].prod(axis=0), axis=0)
          tar_maxs=np.append(tar_maxs,tar.max(axis=0), axis=0)
          tar_mins=np.append(tar_mins,tar.min(axis=0), axis=0)
          tar_means=np.append(tar_means,tar.mean(axis=0), axis=0)
          tar_prods=np.append(tar_prods,tar.prod(axis=0), axis=0)
          
          leftsenti_max=np.append(leftsenti_max,leftsenti.max(axis=0), axis=0)
          rightsenti_max=np.append(rightsenti_max,rightsenti.max(axis=0), axis=0)
          leftsenti_sum=np.append(leftsenti_sum,leftsenti.sum(axis=0), axis=0)
          rightsenti_sum=np.append(rightsenti_sum,rightsenti.sum(axis=0), axis=0)
          
        fullf=fullf.reshape(len(fullf)/size,size)
        leftf=leftf.reshape(len(leftf)/size,size)
        rightf=rightf.reshape(len(rightf)/size,size)
        tarf=tarf.reshape(len(tarf)/size,size)
        fulltw_maxs=fulltw_maxs.reshape(len(fulltw_maxs)/size,size)
        fulltw_mins=fulltw_mins.reshape(len(fulltw_mins)/size,size)
        fulltw_means=fulltw_means.reshape(len(fulltw_means)/size,size)
        fulltw_prods=fulltw_prods.reshape(len(fulltw_prods)/size,size)
        left_maxs=left_maxs.reshape(len(left_maxs)/size,size)
        left_mins=left_mins.reshape(len(left_mins)/size,size)
        left_means=left_means.reshape(len(left_means)/size,size)
        left_prods=left_prods.reshape(len(left_prods)/size,size)
        right_maxs=right_maxs.reshape(len(right_maxs)/size,size)
        right_mins=right_mins.reshape(len(right_mins)/size,size)
        right_means=right_means.reshape(len(right_means)/size,size)
        right_prods=right_prods.reshape(len(right_prods)/size,size)
        tar_maxs=tar_maxs.reshape(len(tar_maxs)/size,size)
        tar_mins=tar_mins.reshape(len(tar_mins)/size,size)
        tar_means=tar_means.reshape(len(tar_means)/size,size)
        tar_prods=tar_prods.reshape(len(tar_prods)/size,size)
        leftsenti_max=leftsenti_max.reshape(len(leftsenti_max)/size,size)
        rightsenti_max=rightsenti_max.reshape(len(rightsenti_max)/size,size)
        leftsenti_sum=leftsenti_sum.reshape(len(leftsenti_sum)/size,size)
        rightsenti_sum=rightsenti_sum.reshape(len(rightsenti_sum)/size,size)
        
        fulltws=np.concatenate([np.median(fulltw_maxs, axis=0),
                                np.median(fulltw_mins, axis=0),
                                np.median(fulltw_means, axis=0),
                                np.std(fullf, axis=0),
                                np.median(fulltw_prods, axis=0)
                                ])
        lefts=np.concatenate([np.median(left_maxs, axis=0),
                              np.median(left_mins, axis=0),
                              np.median(left_means, axis=0),
                              np.std(leftf, axis=0),
                              np.median(left_prods, axis=0)
                              ])
        rights=np.concatenate([np.median(right_maxs, axis=0),
                              np.median(right_mins, axis=0),
                              np.median(right_means, axis=0),
                              np.std(rightf, axis=0),
                              np.median(right_prods, axis=0)
                              ])
        tars=np.concatenate([np.median(tar_maxs, axis=0),
                            np.median(tar_mins, axis=0),
                            np.median(tar_means, axis=0),
                            np.std(tarf, axis=0),
                            np.median(tar_prods, axis=0)
                            ])

        leftsentis=np.concatenate([np.median(leftsenti_max, axis=0),
                                  np.median(leftsenti_sum, axis=0)])
        rightsentis=np.concatenate([np.median(rightsenti_max, axis=0),
                                  np.median(rightsenti_sum, axis=0)])
        if emode == 'full':
            feature=np.concatenate([feature,lefts])
            feature=np.concatenate([feature,rights])
            feature=np.concatenate([feature,tars])
            # feature=np.concatenate([feature,leftsentis])
            # feature=np.concatenate([feature,rightsentis])
                        
        elif emode == 'seg':
            feature=np.concatenate([feature,fulltws])

        return feature
    
    def lidongfeat(self, dataf):
        """ Main function for (Dong et al., 2014) data feature extraction.
        """
        size1=len(self.w2v['the'])
        size2=len(self.sswe['the'])
        data = streamtw(dataf)
        y=[]
        x=np.array([])
        for a, d in enumerate(data):
            feature = np.array([])
            tw = d[0]
            target = d[1]
            y.append(d[2])
            try:
              loc=[i for i, j in enumerate(tw) if j == target]
            except Exception as e:
              print "Couldn't find the tokenised target!"
              print target, tw
            segs, target_position = segment(tw, target)
            feature=self.concattw(feature,size1,segs,'w2v',target_position,target,False,'seg') 
            feature=self.concattw(feature,size2,segs,'sswe',target_position,target,False,'seg')
            # feature=self.concattw(feature,size1,tw,'w2v',loc,target,False,'full') 
            # feature=self.concattw(feature,size2,tw,'sswe',loc,target,False,'full')
            x=np.concatenate([x,feature])
        x=x.reshape((len(y),len(x)/len(y)))
        print x.shape
        return(x,y)   
             
    def elecfeat(self, dataf):
        """ Main function for election data feature extraction.
        """
        size1=len(self.w2v['the'])
        size2=len(self.sswe['the'])
        data = streamtwElec(dataf)
        y=[]
        x=np.array([])
        id = []
        for a, d in enumerate(data):
            feature = np.array([])
            tw = d[0]
            target = d[1]
            if target == '"long_term_economic"_plans':
                target = 'long_term_economic'
            y.append(d[2])
            id.append(d[3])
            whichone = d[4]
            locations = [i for i, j in enumerate(tw) if j == target]
            if (whichone != 'nan') and (len(locations)> 1):
                if whichone >= len(locations):
                    loc = locations[-1]
                else:
                    loc = locations[whichone]
            else:
                loc = tw.index(target)
            segs, target_position = segment(tw, target)
            feature=self.concattw(feature,size1,segs,'w2v',target_position,target,False,'seg') 
            feature=self.concattw(feature,size2,segs,'sswe',target_position,target,False,'seg')
            feature=self.concattw(feature,size1,tw,'w2v',[loc],target,False,'full') 
            feature=self.concattw(feature,size2,tw,'sswe',[loc],target,False,'full')
            x=np.concatenate([x,feature])
        x=x.reshape((len(y),len(x)/len(y)))
        print x.shape
        return(x,y,id)


def main(d):
    features=targettw()
    print "extracting features for training"
    if (d == 'lidong') or (d == 'semeval'):
        x_train,y_train=features.lidongfeat('../data/'+d+'/training/')
    elif d == 'election':
        print 'election training data'
        x_train,y_train,id_train=features.elecfeat('../data/'+d+'/training/')
        writey('../data/'+d+'/output/id_train',id_train)
    writevec('../data/'+d+'/output/training',x_train,y_train)
    
    print "extracting features for testing"
    if (d == 'lidong') or (d == 'semeval'):
        x_test,y_test=features.lidongfeat('../data/'+d+'/testing/')
    elif d == 'election':
        print 'election testing data'
        x_test,y_test,id_test=features.elecfeat('../data/'+d+'/testing/')
    writevec('../data/'+d+'/output/testing',x_test,y_test)
    writey('../data/'+d+'/output/y_test',y_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", dest="d", help="dataset", default='data')
    args = parser.parse_args()
    main(args.d)
    


