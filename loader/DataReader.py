######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import os
import re
import sys
import json
import math
import operator
import random
import itertools
import numpy as np

from FeatParser import *
from DataLexicaliser import *
from utils.nlp import *


class DataReader(object):
    def __init__(self, seed, domain, obj, 
            vocabfile, trainfile, validfile, testfile,
            percentage=1.0, verbose=0, lexCutoff=4):

        self.percentage = percentage # percentage of data used
        # container for data
        self.data   = {'train':[],'valid':[],'test':[]} 
        self.mode   = 'train'     # mode  for accessing data
        self.index  = 0          # index for accessing data
        self.obj    = obj

        # load vocab from file
        self._loadVocab(vocabfile)
        self._loadTokenMap()
        
        ## set input feature cardinality
        self._setCardinality()
       
        ## init formatter/lexicaliser
        self.formatter      = SoftDActFormatter()
        self.lexicaliser    = ExactMatchDataLexicaliser()
        #self.hardformatter  = HardDActFormatter()
        
        ## for lexicalising SLOT_TYPE
        self.lexicaliser.typetoken = domain
        
        # initialise dataset
        self._setupData(trainfile,validfile,testfile)
        #self._testDelexicalisation()

        # obtain pos tags
        #self.obtainTags()
        
        if verbose : self._printStats()
      
    def readall(self,mode='train',processed=False):
        alldata = []
        for feat, dact, sent, base in self.data[mode]:
            a,sv,s,v = self.genFeatVec(feat,self.cardinality,self.dfs)
            alldata.append([a,sv,s,v,sent,dact,base]) 
        return alldata

    def read(self,mode='train',batch=1):
        ## default implementation for read() function
        ## support batch reading & random shuffling
        if self.mode!=mode:
            self.mode = mode
            index = 0

        # end of data , reset index & return None
        if self.index>=len(self.data[mode]):
            data = None
            self.index = 0
            # shuffle data except for testing 
            if mode!='test' : random.shuffle(self.data[mode])
            return data
        
        # reading a batch
        start = self.index
        end   = self.index+batch \
                if self.index+batch<len(self.data[mode])\
                else len(self.data[mode])
        data = self.data[mode][start:end]
        self.index += batch

        # post processing data: a,sv,s,v,sent,dact,base
        proc_data = [[],[],[],[],[],[],[]]
        max_leng  = [0,0,0,0,0]
        for feat,dact,sent,base in data:
            a,sv,s,v = self.genFeatVec(feat,self.cardinality,self.dfs)
            # testing case, 
            if mode=='test':
                sent = [self.lexicalise(sent[i],dact) \
                        for i in range(len(sent))]
                base = [self.lexicalise(base[i],dact) \
                        for i in range(len(base))]
                # put in container
                xvec = [a,sv,s,v,sent,dact,base]
                for j in range(len(proc_data)):
                    proc_data[j].append(xvec[j])
            else:
                # delexicalised text & formatted feature
                if self.obj=='ml': # ML training, 1 sent per example
                    sent = self.delexicalise(sent,dact)
                    sent = self.genInputSent(sent, self.vocab)
                    # put in container
                    xvec = [a,sv,s,v,sent,dact,base]
                    for j in range(len(proc_data)):
                        proc_data[j].append(xvec[j])
                    # padding length
                    for j in range(0,5):
                        if len(xvec[j])>max_leng[j]:
                            max_leng[j] = len(xvec[j])
                else: # TODO:DT training, 1 da/multiple sents per example
                    print a,sv
                    print sent
                    raw_input()
        # padding to have the same sent length
        lengs = [[],[],[],[],[]]
        if mode!='test': # if testing set, not need to process sentence
            for i in range(0,5):
                for j in range(len(proc_data[i])):
                    lengs[i].append(len(proc_data[i][j]))
                    proc_data[i][j].extend([-1]*(max_leng[i]-len(proc_data[i][j])))
        # padding to have the same batch size
        for x in range(end-start,batch):
            for i in range(0,5):
                proc_data[i].append(proc_data[i][-1])
                lengs[i].append(len(proc_data[i][-1]))
        # input/output
        a,sv,s,v =  np.array(proc_data[0],dtype=np.int32),\
                    np.array(proc_data[1],dtype=np.int32),\
                    np.array(proc_data[2],dtype=np.int32),\
                    np.array(proc_data[3],dtype=np.int32)
        if mode!='test':
            words= np.swapaxes(np.array(proc_data[4],dtype=np.int32),0,1)
        else:
            words = proc_data[4]
        dact=proc_data[5]
        base=proc_data[6]
        # current batch size
        b_size = end-start
        lengs  = np.array(lengs,dtype=np.int32)

        return a,sv,s,v, words, dact, base, b_size, lengs

    def delexicalise(self,sent,dact):
        feat = self.formatter.parse(dact,keepValues=True)
        return self.lexicaliser.delexicalise(sent,feat['s2v'])

    def lexicalise(self,sent,dact):
        feat = self.formatter.parse(dact,keepValues=True)
        return self.lexicaliser.lexicalise(sent,feat['s2v'])

    def format(self,dact):
        return self.formatter.format(dact)
    
    def _setCardinality(self):

        self.cardinality = []
        fin = file('resource/feat_template.txt')
        self.dfs = [0,0,0,0,0]
        for line in fin.readlines():
            self.cardinality.append(line.replace('\n',''))
            if line.startswith('a.'):
                self.dfs[1]+=1
            elif line.startswith('sv.'):
                self.dfs[2]+=1
            elif line.startswith('s.'):
                self.dfs[3]+=1
            elif line.startswith('v.'):
                self.dfs[4]+=1
        for i in range(0,len(self.dfs)-1):
            self.dfs[i+1] = self.dfs[i] + self.dfs[i+1]

    def _printStats(self):
        print '==============='
        print 'Data statistics'
        print '==============='
        print 'Train: %d' % len(self.data['train'] )
        print 'Valid: %d' % len(self.data['valid'] )
        print 'Test : %d' % len(self.data['test']  )
        print 'Feat : %d' % len(self.cardinality)
        print '==============='

    def _testDelexicalisation(self):
        for data in self.data['train']+self.data['valid']+self.data['test']:
            dact,sent,base = data
            self.lexicalise(self.delexicalise(sent,dact),dact)

    def _setupData(self,trainfile,validfile,testfile):
        
        # load data from file
        train_group = True if self.obj=='dt' else False
        self.data['train'] = self._loadData(trainfile,train_group)
        self.data['valid'] = self._loadData(validfile,train_group)
        self.data['test' ] = self._loadData(testfile, False, True)
        # cut train/valid data by proportion
        self.data['train'] = self.data['train']\
                [:int(self.percentage*len(self.data['train']))]
        self.data['valid'] = self.data['valid']\
                [:int(self.percentage*len(self.data['valid']))]

    def _loadData(self,filename,group=True,multiref=False):
        
        fin = file(filename)
        # remove comment lines
        for i in range(5):
            fin.readline()
        data = json.load(fin)
        fin.close()
            
        container = []
        for dact,sent,base in data:
            # word tokens
            sent = self.delexicalise( 
                    normalize(re.sub(' [\.\?\!]$','',sent)),dact)
            base = self.delexicalise(
                    normalize(re.sub(' [\.\?\!]$','',base)),dact)
            feat = self.formatter.format(dact)
            container.append( [feat,dact,sent,base] )
        
        # grouping several sentences w/ the same dialogue act
        # for testing set, or DT on train/valid 
        if group or multiref:
            # grouping data points according to unique DAs
            a2ref = {}
            for feat,dact,sent,base in container:
                if a2ref.has_key(tuple(feat)):
                    a2ref[ tuple(feat)][0].append(dact)
                    a2ref[ tuple(feat)][1].append(sent)
                    a2ref[ tuple(feat)][2].append(base)
                else:
                    a2ref[ tuple(feat)] = [[dact],[sent],[base]]
            # return grouped examples
            if group:
                reordered_container = []
                for feat,bundle in a2ref.items():
                    reordered_container.append([feat,
                        bundle[0],bundle[1],bundle[2]])
                return reordered_container 
            # return examples w/ multiple references
            if multiref:
                reordered_container = []
                for feat,dact,sent,base in container:
                    reordered_container.append([feat,dact,
                        a2ref[tuple(feat)][1],
                        a2ref[tuple(feat)][2]])
                return reordered_container
        # if no grouping nor multiref, return directly
        else:   return container
    
    def _loadVocab(self,vocabfile):
        
        fin = file(vocabfile)
        self.vocab = []#['<unk>','</s>']
        for wrd in fin.readlines():
            wrd = wrd.replace('\n','')
            self.vocab.append(wrd)

    def _loadTokenMap(self,mapfile='resource/detect.pair'):
        fin  = file(mapfile)
        self.tokenMap = json.load(fin)['general'].items()
        fin.close()
        # make it 1-to-1 relation
        self.feat2token = {}
        for k,v in self.tokenMap:
            for x in ['1','2','3']:
                key = 'sv.'+k+'._'+x
                self.feat2token[key] = v

    def tokenMap2Indexes(self):
        maxleng = 0
        idxmap = [[] for x in range(len(self.vocab))]
        for k,v in self.feat2token.iteritems():
            try:
                idxmap[self.vocab.index(v)].append(self.cardinality.index(k)-self.dfs[1])
                if len(idxmap[self.vocab.index(v)])>maxleng:
                    maxleng = len(idxmap[self.vocab.index(v)])
            except: pass
        for i in range(len(idxmap)):
            idxmap[i].extend([-1]*abs(len(idxmap[i])-maxleng))
        idxmap = np.array(idxmap,dtype='int32')
        return idxmap

    def readVecFile(self,filename,vocab):
        fin = file(filename)
        # discard comment lines
        for i in range(5):
            fin.readline()
        word2vec = {}
        for line in fin.readlines():
            tokens = line.replace('\n','').split()
            word = tokens[0]
            vec = [float(x) for x in tokens[1:]]
            if word in vocab:
                word2vec[vocab.index(word)] = np.array(vec)
        return word2vec
    
    def genFeatVec(self,feat,cardinality,dfs):
        a,sv,s,v = [],[],[],[]
        a.append(cardinality.index('a.'+feat[0][-1]))
        for item in feat[1:]:
            si,vi = item
            if 'sv.'+si+'.'+vi in cardinality:
                sv.append(  cardinality.index('sv.'+si+'.'+vi)-dfs[1])
            if 's.'+si in cardinality:
                s.append(   cardinality.index('s.'+si)-dfs[2] )
            if 'v.'+vi in cardinality:
                v.append(   cardinality.index('v.'+vi)-dfs[3] )
        if len(feat[1:])==0:
            sv.append(  cardinality.index('sv.NONE.NONE')-dfs[1])
            s.append(   cardinality.index('s.NONE')-dfs[2])
            v.append(   cardinality.index('v.NONE')-dfs[3])
        return a,sv,s,v
        """ 
        featvec = [0.0 for x in range(len(cardinality))]
        featvec[ cardinality.index('a.'+feat[0][-1]) ] = 1.0
        for item in feat[1:]:
            si,vi = item
            featvec[ cardinality.index('s.'+si) ] = 1.0
            featvec[ cardinality.index('v.'+vi) ] = 1.0
            featvec[ cardinality.index('sv.'+si+'.'+vi) ] = 1.0
        if len(feat[1:])==0:
            featvec[ cardinality.index('s.NONE')] = 1.0
            featvec[ cardinality.index('v.NONE')] = 1.0
            featvec[ cardinality.index('sv.NONE.NONE')] = 1.0
        featvec = np.array(featvec,dtype=np.float32)
        return featvec
        """

    def genInputSent(self,sent,vocab):
        words = ['</s>'] + sent.split() + ['</s>']
        wordids = [vocab.index(w) if w in vocab else 0 for w in words]
        return wordids




