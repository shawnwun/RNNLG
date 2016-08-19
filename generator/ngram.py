######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import numpy as np
import os
import operator
from math import sqrt
import random
from ast import literal_eval
from copy import deepcopy

from loader.DataReader import *
from loader.GentScorer import *

from nn.ngmodel import *

from ConfigParser import SafeConfigParser

class Ngram(object):

    def __init__(self,config=None,opts=None):
        # not enough info to execute
        if config==None and opts==None:
            print "Please specify command option or config file ..."
            return
        # config parser
        parser = SafeConfigParser()
        parser.read(config)

        # setting ngram generator parameters
        self.debug      = parser.getboolean('ngram','debug')
        self.seed       = parser.getint('ngram','random_seed')
        self.obj        = 'dt'
        self.trainfile  = parser.get('ngram','train')
        self.validfile  = parser.get('ngram','valid') 
        self.testfile   = parser.get('ngram','test')
        self.vocabfile  = parser.get('ngram','vocab')
        self.domain     = parser.get('ngram','domain')
        self.percentage = float(parser.getfloat('ngram','percentage'))/100.0
        # Setting generation specific parameters
        self.topk       = parser.getint('ngram','topk')
        self.overgen    = parser.getint('ngram','overgen')
        self.beamwidth  = parser.getint('ngram','beamwidth')
        self.detectpairs= parser.get('ngram','detectpairs')
        self.verbose    = parser.getint('ngram','verbose')
        self.N          = parser.getint('ngram','ngram')
        self.rho        = parser.getint('ngram','rho')
        self.decode     = parser.get('ngram','decode')
        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        np.set_printoptions(precision=4)
        # setting data reader, processors, and lexicon
        self.setupSideOperators()
        
    def testNgram(self):

        ######## train ngram generator by grouping ########
        if self.debug:
            print 'start ngram training ...'
        da2sents = {}
        templates = self.reader.readall(mode='train')+\
                    self.reader.readall(mode='valid')
        for a,sv,s,v,sents,dact,base in templates:
            key = (tuple(a),tuple(sv))
            if da2sents.has_key(key):
                da2sents[key].extend(sents)
                da2sents[key] = list(set(da2sents[key]))
            else:
                da2sents[key] = sents
        # accumulate texts for training class-based LM
        cls2texts = {}
        for key,sents in da2sents.iteritems():
            a,sv = key
            identifier = (a,sv[:self.rho]) if len(sv)>self.rho else (a,sv)
            if cls2texts.has_key(identifier):
                cls2texts[identifier].extend(sents)
            else:
                cls2texts[identifier] = sents
        
        # train class based ngram models
        cls2model = {}
        for key, sents in cls2texts.iteritems():
            model = NGModel(self.reader.vocab,self.topk,self.overgen,
                    self.beamwidth,n=self.N,rho=self.rho)
            model.train(sents)
            cls2model[key] = model

        ######## test ngram generator on test set ######### 
        if self.debug:
            print 'start ngram generation ...'
        
        # container
        parallel_corpus, hdc_corpus = [], []
        # slot error counts
        gencnts, refcnts = [0.0,0.0,0.0],[0.0,0.0,0.0]

        while True:
            # read data point
            data = self.reader.read(mode='test',batch=1)
            if data==None:
                break
            a,sv,s,v,sents,dact,bases,cutoff_b,cutoff_f = data
            # remove batch dimension
            a,sv,s,v = a[0],sv[0],s[0],v[0]
            sents,dact,bases = sents[0],dact[0],bases[0]
            # score DA similarity between testing example and class LMs
            model_ranks = []
            for da_t,model in cls2model.iteritems():
                a_t,sv_t = [set(x) for x in da_t]
                # cosine similarity
                score =float(len(a_t.intersection(set(a)))+\
                        len(sv_t.intersection(set(sv))))/\
                        sqrt(len(a_t)+len(sv_t))/sqrt(len(a)+len(sv))
                model_ranks.append([score,model])
            # rank models
            model_ranks = sorted(model_ranks,key=operator.itemgetter(0))
            score,model = model_ranks[-1]
            # sample or beam search
            if self.decode=='sample':
                gens = model.sample()
            elif self.decode=='beam':
                gens = model.beamSearch()
            # for slot error rate scoring
            felements = [self.reader.cardinality[x+self.reader.dfs[1]]\
                    for x in sv]
            
            # post processing
            for i in range(len(gens)):
                penalty, gen = gens[i]
                # replace word id with actual words
                gen = ' '.join([x for x in gen[1:-1]])
                # score slot error rate
                cnt, total, caty = self.gentscorer.scoreERR(a,felements,gen)
                # update score by categorical slot errors
                penalty += caty
                # lexicalise back
                gens[i] = (penalty,self.reader.lexicalise(gen,dact))
            # get the top-k for evaluation
            gens = sorted(gens,key=operator.itemgetter(0))[:self.topk]
            # print results
            print dact
            print 'Penalty\tTSER\tASER\tGen'
            for penalty, gen in gens:
                # score slot error rate
                cnt, total, caty = self.gentscorer.scoreERR(a,felements,
                        self.reader.delexicalise(gen,dact))
                # accumulate slot error cnts
                gencnts[0]  += cnt
                gencnts[1]  += total
                gencnts[2]  += caty
                print '%.4f\t%d\t%d\t%s' % (penalty,total,caty,gen)
            print '\n'
            # compute gold standard slot error rate
            for sent in sents:
                # score slot error rate
                cnt, total, caty = self.gentscorer.scoreERR(a,felements,
                        self.reader.delexicalise(sent,dact))
                # accumulate slot error cnts
                refcnts[0]  += cnt
                refcnts[1]  += total
                refcnts[2]  += caty
            # accumulate score for bleu score computation         
            parallel_corpus.append([[g for s,g in gens],sents])
            hdc_corpus.append([bases[:1],sents])

        bleuModel   = self.gentscorer.scoreBLEU(parallel_corpus)
        bleuHDC     = self.gentscorer.scoreBLEU(hdc_corpus)
        print '##############################################'
        print 'BLEU SCORE & SLOT ERROR on GENERATED SENTENCES'
        print '##############################################'
        print 'Metric       :\tBLEU\tT.ERR\tA.ERR'
        print 'HDC          :\t%.4f\t%2.2f%%\t%2.2f%%'% (bleuHDC,0.0,0.0)
        print 'Ref          :\t%.4f\t%2.2f%%\t%2.2f%%'% (1.0,
                100*refcnts[1]/refcnts[0],100*refcnts[2]/refcnts[0])
        print '----------------------------------------------'
        print 'This Model   :\t%.4f\t%2.2f%%\t%2.2f%%'% (bleuModel,
                100*gencnts[1]/gencnts[0],100*gencnts[2]/gencnts[0])

    def setupSideOperators(self):
        # initialise data reader
        self.reader = DataReader(self.seed, self.domain, self.obj,
                self.vocabfile, self.trainfile, self.validfile, self.testfile,
                self.percentage, self.verbose, lexCutoff=4)
        # setting generation scorer
        self.gentscorer = GentScorer(self.detectpairs)
        


