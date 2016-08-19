######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import operator
import numpy as np
from Queue import PriorityQueue
from utils.mathUtil import softmax, sigmoid, tanh
from copy import deepcopy
from basic import *

class NGModel(object):

    def __init__(self, vocab, topk, overgen, beamwidth, n=5, rho=3):
        self.n          = n
        self.rho        = rho
        self.vocab      = vocab
        self.topk       = topk
        self.overgen    = overgen
        self.beamwidth  = beamwidth

    def train(self,sents):

        # calculate the max/min length of sentences
        lengs = [len(sent.split())+2 for sent in sents]
        self.maxleng = max(lengs)
        self.minleng = min(lengs)
      
        # start ngram training
        self.model = dict([(n,{}) for n in range(1,self.n+1)])
        self.model[1] = {}
        for sent in sents:
            words = sent.split()
            words = ['</s>'] + words + ['</s>']
            # unigram
            for w in words:
                if self.model[1].has_key(w):
                    self.model[1][w] += 1.0
                else:
                    self.model[1][w] =  1.0
            # for >2 grams
            for n in range(2,self.n+1):
                # scan through sentence
                for i in range(n-1,len(words)):
                    # context
                    context = tuple(words[i-n+1:i])
                    word = words[i]
                    # find context
                    if self.model[n].has_key(context):
                        # seen context and word
                        if self.model[n][context].has_key(word):
                            self.model[n][context][word]+=1.0
                        else: # new word
                            self.model[n][context][word] =1.0
                    else: # not found, init new context dict
                        self.model[n][context] = {word:1.0}

        # normalise everything into log prob
        sum1 = sum([cnt for cnt in self.model[1].values()])
        for w,cnt in self.model[1].iteritems():
            self.model[1][w] = cnt/sum1
        self.model[1] = sorted(self.model[1].items(),
                key=operator.itemgetter(1),reverse=True)
        for n in range(2,self.n+1):
            for context,wdct in self.model[n].iteritems():
                sumc = sum([cnt for cnt in wdct.values()])
                for w,cnt in wdct.iteritems():
                    self.model[n][context][w] = cnt/sumc
                self.model[n][context] = sorted(
                        self.model[n][context].items(),
                        key=operator.itemgetter(1),reverse=True)
        
    def beamSearch(self):
        # end nodes
        endnodes = []
        # starting node
        node = BeamSearchNode(None,None,None,'</s>',0,1)
        # queue for beam search
        nodes= PriorityQueue()
        nodes.put((-node.eval(),node))
        qsize = 1
        # start beam search
        while True:
            # give up when decoding takes too long 
            if qsize>10000 or nodes.empty(): break
            # fetch the best node
            score, n = nodes.get()
            # if end of sentence token 
            if n.wordid.endswith('</s>') and n.prevNode!=None:
                endnodes.append((score,n))
                # if reach maximum # of sentences required
                if len(endnodes)>=self.overgen: break
                else:                           continue
            # decode for one step using decoder 
            words, probs = self._gen(n)
            # put them into a queue
            for i in range(len(words)):
                node = BeamSearchNode(None,None,n,
                        n.wordid+' '+words[i],
                        n.logp+np.log10(probs[i]),n.leng+1)
                nodes.put( (-node.eval(),node) )
            # increase qsize
            qsize += len(words)-1
        # if no finished nodes, choose the top scored paths
        if len(endnodes)==0:
            endnodes = [nodes.get() for n in range(self.overgen)]
        # choose nbest paths, back trace them
        utts = []
        for penalty,n in sorted(endnodes,key=operator.itemgetter(0)):
            utt = n.wordid.split()
            # penalise length
            penalty = penalty+0.1*abs(len(utt)-self.maxleng) if \
                    len(utt)>self.maxleng else penalty
            penalty = penalty+0.1*abs(len(utt)-self.minleng) if \
                    len(utt)<self.minleng else penalty
            utts.append((penalty,utt))
        return utts

    def sample(self):
        # container
        gens = []
        # to obtain topk generations
        for i in range(self.overgen):
            # starting node
            node = BeamSearchNode(None,None,None,'</s>',0,1)
            # put in queue
            nodes = [node]
            # start sampling
            while True:
                # check stopping criteria
                last_node = nodes[-1]
                if last_node.wordid=='</s>' and len(nodes)>1:
                    break
                if len(nodes)>40: # undesirable long utt
                    break
                # expand for one time step
                words, probs = self._gen(last_node)
                # sampling according to probability
                o_sample = np.argmax(np.random.multinomial(1,probs,1))
                # put new node into the queue
                node = BeamSearchNode(None,None,last_node,words[o_sample],
                        last_node.logp+np.log10(probs[o_sample]),last_node.leng+1)
                nodes.append( node )
            # obtain sentence
            gen = [n.wordid for n in nodes]
            # score the sentences
            score = -nodes[-1].eval()
            # make sure the generated sentence doesn't repeat
            if [score,gen] not in gens:
                gens.append([score,gen])
        # ranking generation according to score
        overgen = self.overgen if len(gens)>self.overgen else len(gens)
        for i in range(len(gens)):
            penalty, gen = gens[i]
            # penalise length
            penalty = penalty+0.1*abs(len(gen)-self.maxleng) if \
                    len(gen)>self.maxleng else penalty
            penalty = penalty+0.1*abs(len(gen)-self.minleng) if \
                    len(gen)<self.minleng else penalty
            gens[i][0] = penalty
        gens = sorted(gens,key=operator.itemgetter(0))[:overgen]
        return gens 

    def _gen(self,node):
       
        # split context
        context = node.wordid.split()

        # consider which N of n-grams to generate
        NG = len(context)+1 if len(context)+1<=self.n else self.n
        words, probs = [],[]
        while NG>1: # while >1 gram 
            picked_ngram = self.model[NG]
            cntxt = tuple(context[-NG+1:])
            if self.model[NG].has_key(cntxt) and len(self.model[NG][cntxt])>0:
                # if cannot find the context using current ngram
                cutoff = self.beamwidth if \
                        len(self.model[NG][cntxt])>self.beamwidth\
                        else len(self.model[NG][cntxt])
                words,probs = map(list, zip(*self.model[NG][cntxt][:cutoff]))
                return words, probs
            else:# fallback to n-1 gram
                NG-=1
        # choose from unigram
        cutoff = self.beamwidth if len(self.model[1])>self.beamwidth\
                else len(self.model[1])
        words,probs = map(list, zip(*self.model[1][:cutoff]))
        return words,probs 

