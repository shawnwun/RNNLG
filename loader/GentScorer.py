######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import os
import json
import sys
import math
import operator

from collections import Counter
from nltk.util import ngrams
#from nltk.align.bleu import BLEU


class ERRScorer():

    ## Scorer for calculating the slot errors
    ## it scores utterances one by one
    ## using two levels of matching 
    ## 1. exact match for categorical values
    ## 2. multiple keyword matching for binary values
    ## 3. cannot deal with don't care and none values
    def __init__(self,detectfile):

        self.detectPairs = []
        fin = file(detectfile)
        self.detectPairs = json.load(fin)
        fin.close()

    def countSlots(self,dataset,reader):
        count = 0
        for t in dataset:
            feat = reader.formatter.format(t[0])[0]
            c = count
            for s,v in feat:
                # skip type token
                if s=='type':
                    continue
                if v=='_' or v=='yes' or v=='none' or v=='no':
                    count +=1
        return count

    def score(self,a,feat,gen):
        # total slots
        slot_count = 0
        # exact match for categorical slots
        caty_slot_error = 0
        # fo each slot - token pair in the detect pair dict
        for s,tok in self.detectPairs['general'].iteritems(): 
            # token compare to
            comparetos = ['sv.'+s+'._1','sv.'+s+'._2','sv.'+s+'._3']
            # count feature count in da feature
            fcnt = 0
            for f in feat:
                for compareto in comparetos:
                    if compareto==f:  fcnt+=1
            # count generated semantic tokens
            gcnt = gen.split().count(tok)
            # count the slot difference
            #if fcnt!=gcnt:
            #    caty_slot_error += 1.0
            caty_slot_error += abs(fcnt-gcnt)
            # accumulate slot count
            slot_count += fcnt

        # key word match for binary slots, only an approximation
        bnay_slot_error = 0
        # for each binary slot
        for s,toks in self.detectPairs['binary'].iteritems():
            # tokens compare to
            comparetos = ['sv.'+s+'.yes','sv.'+s+'.no',
                    'sv.'+s+'.dontcare','sv.'+s+'.none']
            # count feature occurrence in da
            fcnt = 0
            for f in feat:
                for compareto in comparetos:
                    if compareto==f:  fcnt+=1
            # count generated semantic tokens
            gcnt = sum([gen.split().count(tok) for tok in toks]) 
            # count the slot difference
            bnay_slot_error += abs(fcnt-gcnt)
            # accumulate slot count
            slot_count += fcnt
        # total slot error
        total_slot_error = caty_slot_error + bnay_slot_error
        # when ?select/suggest act, only consider categorical errors
        if a==[4] or a==[14]:
            #return slot_count, caty_slot_error, caty_slot_error
            return 0.0,0.0,0.0
        else:
            return slot_count, total_slot_error, caty_slot_error

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass
    def score(self,parallel_corpus):
        
        # containers and parameters
        r,c = 0,0
        count = [0,0,0,0]
        clip_count = [0,0,0,0]
        weights=[0.25,0.25,0.25,0.25]

        # accumulate ngram statistics
        for hyps,refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # compute ngram counts by matching each hypothesis
            for hyp in hyps:
                # for each ngram
                for i in range(4):
                    # accumulate hyp ngram counts
                    hypcnts = Counter(ngrams(hyp,i+1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts 
                    max_counts = {}
                    # compare to each reference
                    for ref in refs:
                        # get reference ngrams
                        refcnts = Counter(ngrams(ref, i+1))
                        # for each ngram
                        for ng in hypcnts:
                            # clipped counts
                            max_counts[ng] = max( max_counts.get(ng,0),refcnts[ng] )
                    # compute clipped counts by clipping the hyp count if necessary
                    clipcnt = dict( (ng,min(count,max_counts[ng])) \
                            for ng,count in hypcnts.items() )
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c, find best match among all references
                bestmatch = [1000,1000]
                for ref in refs:
                    if bestmatch[0]==0: break
                    # length difference
                    diff = abs(len(ref)-len(hyp))
                    # if the current diff less than stored one, change it
                    if diff<bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                # extract the best length match in references
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        # for numerical stability
        p0 = 1e-7
        # brevity penality
        bp = 1 if c>r else math.exp(1-float(r)/float(c))
        # modified prec.
        p_ns = [float(clip_count[i])/float(count[i]+p0)+p0 \
                for i in range(4)]
        # weighted prec.
        s = math.fsum(w*math.log(p_n) \
                for w, p_n in zip(weights, p_ns) if p_n)
        # final bleu score
        bleu = bp*math.exp(s)
        return bleu

    def sentence_bleu_4(self,parallel_corpus):
        # input : single sentence, multiple references
        count = [0,0,0,0]
        clip_count = [0,0,0,0]
        weights=[0.25,0.25,0.25,0.25]
        r = 0
        c = 0
        
        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # compute ngram counts by matching each hypothesis
            for hyp in hyps:
                # for each ngram
                for i in range(4):
                    # accumulate hyp ngram counts
                    hypcnts = Counter(ngrams(hyp,i+1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts 
                    max_counts = {}
                    # compare to each reference
                    for ref in refs:
                        # get reference ngrams
                        refcnts = Counter(ngrams(ref, i+1))
                        # for each ngram
                        for ng in hypcnts:
                            # clipped counts
                            max_counts[ng] = max( max_counts.get(ng,0),refcnts[ng] )
                    # compute clipped counts by clipping the hyp count if necessary
                    clipcnt = dict( (ng,min(count,max_counts[ng])) \
                            for ng,count in hypcnts.items() )
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c, find best match among all references
                bestmatch = [1000,1000]
                for ref in refs:
                    if bestmatch[0]==0: break
                    # length difference
                    diff = abs(len(ref)-len(hyp))
                    # if the current diff less than stored one, change it
                    if diff<bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                # extract the best length match in references
                r += bestmatch[1]
                c += len(hyp)
        
        # for numerical stability
        p0 = 1e-7
        # modified brevity penality
        bp = math.exp(-abs(1.0-float(r)/float(c+p0)))
        # smoothed version of modified prec.
        p_ns = [0,0,0,0]
        for i in range(4):
            if i<2: # original version n-gram counts
                p_ns[i] = float(clip_count[i])/float(count[i]+p0)+p0
            else: # smoothed version of ngram counts
                smooth_term = 5*p_ns[i-1]*p_ns[i-1]/p_ns[i-2]
                p_ns[i] = float(clip_count[i]+smooth_term)/float(count[i]+5)+p0
        # weighted prec.
        s = math.fsum(w*math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
        # final sentence bleu score
        bleu_hyp = bp*math.exp(s)
        return bleu_hyp


class GentScorer(object):
    ## main Scorer interfaces for all scorers
    ## it can do 
    ## 1. Compute bleu score
    ## 2. Compute slot error rate
    ## 3. Detailed illustraction of how differet split 
    ##    of data affect performance
    def __init__(self,detectfile):
        self.errscorer = ERRScorer(detectfile)
        self.bleuscorer= BLEUScorer()

    def scoreERR( self,a,feat,gen):
        return self.errscorer.score(a,feat,gen)
        
    def countSlots(self,dataset,reader):
        return self.errscorer.countSlots(dataset,reader)

    def scoreBLEU(self,parallel_corpus):
        return self.bleuscorer.score(parallel_corpus)

    def scoreSBLEU(self,parallel_corpus):
        return self.bleuscorer.sentence_bleu_4(parallel_corpus)


