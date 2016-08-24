######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import sys
import numpy as np
import theano.tensor as T
import theano.gradient as G
from collections import OrderedDict

from basic  import *
from hlstm  import *
from sclstm import *
from encdec import *

class NNGenerator(object):

    def __init__(self, gentype, vocab, beamwidth, overgen,
            vocab_size, hidden_size, batch_size, feat_sizes, 
            obj='ml', train_mode='all', decode='beam',tokmap=None):

        # hyperparameters
        self.gentype= gentype
        self.decode = decode
        self.mode   = train_mode
        self.di = vocab_size
        self.dh = hidden_size
        self.db = batch_size
        self.dfs= feat_sizes
        self.obj= obj
        
        # choose generator architecture
        self.params = []
        if self.gentype=='sclstm':
            self.generator = sclstm(self.gentype,vocab,
                    beamwidth,overgen,
                    self.di,self.dh,self.db,self.dfs)
            self.params = self.generator.params
        elif self.gentype=='encdec':
            self.generator = encdec(self.gentype,vocab,
                    beamwidth,overgen,
                    self.di,self.dh,self.db,self.dfs)
            self.params = self.generator.params
        elif self.gentype=='hlstm':
            self.generator = hlstm(self.gentype,vocab,
                    beamwidth,overgen,
                    self.di,self.dh,self.db,self.dfs,
                    tokmap)
            self.params = self.generator.params
   
    def config_theano(self):
        # input tensor variables
        w_idxes = T.imatrix('w_idxes')
        w_idxes = T.imatrix('w_idxes')
        a       = T.imatrix('a')
        sv      = T.imatrix('sv')
        s       = T.imatrix('s')
        v       = T.imatrix('v')
        
        # cutoff for batch and time
        cutoff_f  = T.imatrix('cutoff_f')
        cutoff_b  = T.iscalar('cutoff_b')
        
        # regularization and learning rate
        lr   = T.scalar('lr')
        reg  = T.scalar('reg')

        # unroll generator and produce cost
        if self.gentype=='sclstm':
            self.cost, cutoff_logp = \
                    self.generator.unroll(a,sv,w_idxes,cutoff_f,cutoff_b)
        elif self.gentype=='encdec':
            self.cost, cutoff_logp = \
                    self.generator.unroll(a,s,v,w_idxes,cutoff_f,cutoff_b)
        elif self.gentype=='hlstm':
            self.cost, cutoff_logp = \
                    self.generator.unroll(a,sv,w_idxes,cutoff_f,cutoff_b)
        
        ###################### ML Training #####################
        # gradients and updates
        gradients = T.grad( clip_gradient(self.cost,1),self.params )
        updates = OrderedDict(( p, p-lr*g+reg*p ) \
                for p, g in zip( self.params , gradients))

        # theano functions
        self.train = theano.function(
                inputs= [a,sv,s,v, w_idxes, cutoff_f, cutoff_b, lr, reg],
                outputs=-self.cost,
                updates=updates,
                on_unused_input='ignore') 
        self.test  = theano.function(
                inputs= [a,sv,s,v, w_idxes, cutoff_f, cutoff_b],
                outputs=-self.cost,
                on_unused_input='ignore')
        
        ###################### DT Training #####################
        # expected objective
        bleus   = T.fvector('bleu')
        errs    = T.fvector('err')
        gamma   = T.iscalar('gamma')

        senp  = T.pow(10,gamma*cutoff_logp/cutoff_f[4][:cutoff_b])/\
                T.sum(T.pow(10,gamma*cutoff_logp/cutoff_f[4][:cutoff_b]))
        xBLEU = T.sum(senp*bleus[:cutoff_b])
        xERR  = T.sum(senp*errs[:cutoff_b])
        self.obj = -xBLEU + 0.3*xERR
        obj_grad = T.grad( clip_gradient(self.obj,1),self.params )
        obj_updates = OrderedDict(( p, p-lr*g+reg*p ) \
                for p, g in zip( self.params , obj_grad))

        # expected objective functions
        self.trainObj = theano.function(
                inputs= [a,sv,s,v, w_idxes, cutoff_f, cutoff_b,
                    bleus, errs, gamma, lr, reg],
                outputs=[self.obj,xBLEU,xERR,senp],
                updates=obj_updates,
                on_unused_input='ignore',
                allow_input_downcast=True)
        self.testObj = theano.function(
                inputs= [a,sv,s,v, w_idxes, cutoff_f, cutoff_b,
                    bleus,errs,gamma],
                outputs=[self.obj,xBLEU,xERR],
                on_unused_input='ignore',
                allow_input_downcast=True)

    def gen(self,a,sv,s,v):
        if self.decode=='beam':
            if self.gentype=='sclstm':
                return self.generator.beamSearch(a,sv)
            elif self.gentype=='encdec':
                return self.generator.beamSearch(a,s,v)
            elif self.gentype=='hlstm':
                return self.generator.beamSearch(a,sv)
        else:
            if self.gentype=='sclstm':
                return self.generator.sample(a,sv)
            elif self.gentype=='encdec':
                return self.generator.sample(a,s,v)
            elif self.gentype=='hlstm':
                return self.generator.sample(a,sv)

    def setWordVec(self,word2vec):
        self.generator.setWordVec(word2vec)

    def setParams(self,params):
        for i in range(len(self.params)):
            self.params[i].set_value(params[i])

    def getParams(self):
        return [p.get_value() for p in self.params]

    def numOfParams(self):
        return sum([p.get_value().size for p in self.params])

    def loadConverseParams(self):
        self.generator.loadConverseParams()

