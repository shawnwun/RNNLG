######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import numpy as np
import theano
import theano.tensor as T

# numerical stability
eps = 1e-7

# gradient clipping 
class GradClip(theano.compile.ViewOp):
    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]

def clip_gradient(x, bound):
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)

# obtain sent logprob by summing over word logprob
def collectSentLogp(p,cutoff_t,cutoff_b):
    q = p.dimshuffle(1,0)
    def sump(p_b,stop_t):
        logp = T.sum(T.log10(p_b[:stop_t]))
        return logp
    cutoff_logp, _ = theano.scan(fn=sump,\
        sequences=[q[:cutoff_b],cutoff_t[:cutoff_b]],\
        outputs_info=[None])
    return cutoff_logp

# Node class for performing beam search
class BeamSearchNode(object):

    def __init__(self,h,c,prevNode,wordid,logp,leng):
        self.h      = h
        self.c      = c
        self.logp   = logp
        self.leng   = leng
        self.wordid = wordid
        self.prevNode = prevNode
        self.sv = None
    
    def eval(self):
        if self.leng>40:
            return self.logp/float(self.leng-1+eps)-40.0
        return self.logp/float(self.leng-1+eps)

# basic class for Recurrent Language Generator
class BaseRLG(object):

    def __init__(self, gentype, beamwidth, overgen,
            vocab_size, hidden_size, batch_size, da_sizes):

        # setting hyperparameters
        self.gentype= gentype
        self.di     = vocab_size
        self.dh     = hidden_size
        self.db     = batch_size
        self.dfs    = da_sizes
        self.overgen= overgen
        self.beamwidth = beamwidth

    def _init_params(self):
        #TODO: function for initialise weight matrices
        pass

    def unroll(self):
        #TODO: unrolling function in theano, for training
        pass

    def _recur(self):
        #TODO: per step recurrence function in theano, for training
        pass

    def beamSearch(self):
        #TODO: generation function in numpy, beam search decoding
        pass

    def sample(self):
        #TODO: generation function in numpy, random sampling
        pass

    def _gen(self):
        #TODO: per step generation function in numpy, for decoding
        pass
    
    def loadConverseParams(self):
        #TODO: load numpy parameters
        pass
   
    def setParams(self,params):
        # set theano parameters
        for i in range(len(self.params)):
            self.params[i].set_value(params[i])

    def getParams(self):
        # fetch theano parameters
        return [p.get_value() for p in self.params]

    def numOfParams(self):
        # return number of parameters
        return sum([p.get_value().size for p in self.params])
 
