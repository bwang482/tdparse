import os
import re
import codecs
from ftfy import fix_text
from gensim import utils
from twtokenize import tokenize


class streamtw(object):
    """Iterate over sentences from (Dong et al., 2014) dataset"""
    def __init__(self, dirname):
        self.i=0
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):

            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                self.i+=1
                if self.i==1:
                    tw=line.lower().strip()
                if self.i==2:
                    target=line.lower().strip()
                if self.i==3:
                    senti=int(line.strip())+1
                    tw=tw.replace(target,' '+target+' ')
                    tw=tw.replace(''.join(target.split()),' '+'_'.join(target.split())+' ')
                    tw=tw.replace(target,' '+'_'.join(target.split())+' ')
                    tweet=tokenize(tw)
                    yield (tweet,'_'.join(target.split()),senti)
                    self.i=0

#----------------------------------------------------------------------------

class streamtwElec(object):
    """Iterate over sentences from the election dataset"""
    def __init__(self, dirname):
        self.i=0
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = line.split('\t')
                id = line[1]
                if line[2] == 'positive':
                    sent = 1
                elif line[2] == 'negative':
                    sent = -1
                elif line[2] == 'neutral':
                    sent = 0
                senti = sent + 1
                target = line[3].lower().strip()
                location = line[4]
                tw = line[-1].lower().strip()
                tw = fix_text(tw.decode('utf-8'))
                range = []
                p = re.compile(r'(?<!\w)({0})(?!\w)'.format(target))
                for m in p.finditer(tw.lower()):
                      range.append([m.start(),m.start()+len(m.group())])
                if location != 'nan':
                   cc = 0
                   for a, b in enumerate(range):
                       if b[0]-1 <= int(location) <= b[1]+4:
                          wh = a
                          cc=1
                   if cc==0:
                       wh = 'nan'
                else:
                    wh = location  
                if wh == 'nan':
                    tw=tw.replace(target,' '+target+' ')
                    tw=tw.replace(''.join(target.split()),' '+'_'.join(target.split())+' ')
                    tw=tw.replace(target,' '+'_'.join(target.split())+' ')  
                else:
                    try:
                        r = range[wh]
                    except:
                        print "Error at processing election data; at line 85 process_data.py!"
                    tw=tw[:r[0]]+ tw[r[0]:r[1]+2].replace(target, ' '+target+' ') + tw[r[1]+2:]
                    tw=tw[:r[0]]+ tw[r[0]:r[1]+4].replace(''.join(target.split()),' '+'_'.join(target.split())+' ') + tw[r[1]+4:]
                    tw=tw[:r[0]]+ tw[r[0]:r[1]+6].replace(target,' '+'_'.join(target.split())+' ') + tw[r[1]+6:]              
                tweet=tokenize(tw)
                yield (tweet,'_'.join(target.split()),senti,id,wh)

#------------------------------------------------------------------------------

class streampos(object):
    """Iterate over part-of-speech tags"""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                tw,pos,score,tokens=line.strip().split('\t')
                yield pos


