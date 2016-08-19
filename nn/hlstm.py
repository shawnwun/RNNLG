######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import operator
import numpy as np
import theano.tensor as T
from Queue import PriorityQueue
from utils.mathUtil import softmax, sigmoid, tanh
from theano.ifelse import ifelse
from copy import deepcopy

from basic  import *

class hlstm(BaseRLG):

    def __init__(self, gentype, vocab, beamwidth, overgen,
            vocab_size, hidden_size, batch_size, da_sizes,
            tokmap):
        
        # calling superclass constructor
        BaseRLG.__init__(self, gentype, beamwidth, overgen,
                vocab_size, hidden_size, batch_size, da_sizes)
        self.dsv    = self.dfs[2]-self.dfs[1]
        self.da     = self.dfs[1]-self.dfs[0]
        self.vocab  = vocab
        # save mapping dict
        self.tokmap     = theano.shared(tokmap)
        self.tokmap_np  = tokmap
        # init params
        self._init_params()

    def _init_params(self):
        
        # word embedding weight matrix
        self.Wemb   = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.di,self.dh)).astype(theano.config.floatX))
        # lstm gate weight matrix
        self.Wgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*3,self.dh*4)).astype(theano.config.floatX))
        # 1hot DA to distributed vector weight matrix
        self.Wfc= theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.da+self.dsv,self.dh)).astype(theano.config.floatX))
        # hidden to output weight matrix
        self.Who= theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh,self.di)).astype(theano.config.floatX))
        # initial memory cell and hidden layer
        self.h0 = theano.shared(np.zeros((self.db,self.dh),
            dtype=theano.config.floatX))
        self.c0 = theano.shared(np.zeros((self.db,self.dh),
            dtype=theano.config.floatX))
        # all parameters
        self.params = [ 
                self.Wemb, 
                self.Wgate, 
                self.Wfc,
                self.Who ]
    
    def setWordVec(self,word2vec):
        self.Wemb_np = self.Wemb.get_value()
        for w,v in word2vec.iteritems():
            self.Wemb_np[w,:] = v
        self.Wemb.set_value(self.Wemb_np)
   
    # form 1-hot vector
    def _form1hot(self, hot_x, idx_x, cutoff_x):
        update_x = T.set_subtensor(hot_x[idx_x[:cutoff_x]],1.0)
        return update_x
    
    # align for each batch
    def _batchAlign(self, w_tb, mask_b):
        mask_b = T.set_subtensor(mask_b[self.tokmap[w_tb]],0)
        return mask_b 
    
    # heuristic alignment 
    def _alignData(self, w_t, sv_tm1):
        # padding dummy element 
        mask = T.concatenate([T.ones_like(sv_tm1),
            T.ones_like(sv_tm1[:,-1:])],axis=1)
        # iterate over batch 
        mask,_ = theano.scan(fn=self._batchAlign,
                sequences=[w_t,mask],
                outputs_info=None)
        # mask the slot-value vector
        sv_t = mask[:,:-1] * sv_tm1
        return sv_t

    def unroll(self,a,sv,words,cutoff_f,cutoff_b):

        # form 1-hot representation
        a_1hot = theano.shared(np.zeros((self.db,self.da),
            dtype=theano.config.floatX))
        sv_1hot= theano.shared(np.zeros((self.db,self.dsv),
            dtype=theano.config.floatX))
        a_1hot ,_= theano.scan(fn=self._form1hot,
                sequences=[a_1hot,a,cutoff_f[0]])
        sv_1hot,_= theano.scan(fn=self._form1hot,
                sequences=[sv_1hot,sv,cutoff_f[1]])

        # heuristically align dataset
        aligned_sv,_ = theano.scan(fn=self._alignData,
                sequences=[words[:-1,:]],
                outputs_info=[sv_1hot],
                non_sequences=None)

        # recurrence
        [h,c,p],_ = theano.scan(fn=self._recur,
                sequences=[words[:-1,:],words[1:,:],aligned_sv],
                outputs_info=[self.h0,self.c0,None],
                non_sequences=[a_1hot])
        # compute desired sent_logp by slicing
        cutoff_logp = collectSentLogp(p,cutoff_f[4],cutoff_b)
        cost = -T.sum(cutoff_logp) 
        return cost, cutoff_logp

    def _recur(self, w_t, y_t, sv_tm1, h_tm1, c_tm1, a):
        
        # input word embedding
        wv_t = T.nnet.sigmoid(self.Wemb[w_t,:])
        # encode da
        da_emb_t = T.tanh(T.dot(T.concatenate([a,sv_tm1],axis=1),self.Wfc))
        # compute ig, fg, og together and slice it
        gates_t = T.dot( T.concatenate([wv_t,h_tm1,da_emb_t],axis=1),self.Wgate)
        ig  = T.nnet.sigmoid(gates_t[:,:self.dh])
        fg  = T.nnet.sigmoid(gates_t[:,self.dh:self.dh*2])
        og  = T.nnet.sigmoid(gates_t[:,self.dh*2:self.dh*3])
        cx_t= T.tanh(gates_t[:,self.dh*3:])
        # update lstm internal state
        c_t = ig*cx_t + fg*c_tm1 
        # obtain new hiddne layer
        h_t = og*T.tanh(c_t)
        # compute output distribution target word prob
        o_t = T.nnet.softmax( T.dot(h_t,self.Who) )
        p_t = o_t[T.arange(self.db),y_t]

        return h_t, c_t, p_t

    def _get1hot(self,idxes,dim):
        vec = np.zeros(dim)
        vec[idxes] = 1.0
        return vec

    def beamSearch(self,a,sv):
        # get 1 hot vector
        a = self._get1hot(a,self.da)
        sv= self._get1hot(sv,self.dsv)
        # end nodes
        endnodes = []
        # initial layers
        h0,c0 = np.zeros(self.dh),np.zeros(self.dh)
        # starting node
        node = BeamSearchNode(h0,c0,None,1,0,1)
        node.sv = sv
        node.a  = a
        # queue for beam search
        nodes= PriorityQueue()
        nodes.put((-node.eval(),node))
        qsize = 1
        # start beam search
        while True:
            # give up when decoding takes too long 
            if qsize>10000: break
            # fetch the best node
            score, n = nodes.get()
            # if end of sentence token 
            if n.wordid==1 and n.prevNode!=None:
                endnodes.append((score,n))
                # if reach maximum # of sentences required
                if len(endnodes)>=self.overgen: break
                else:                           continue
            # decode for one step using decoder 
            words, probs, c, h = self._gen(n)
            # put them into a queue
            for i in range(len(words)):
                node = BeamSearchNode(h,c,n,words[i],
                        n.logp+np.log10(probs[i]),n.leng+1)
                # heuristically update DA
                node.sv = n.sv
                node.a  = n.a
                tmpsv   = np.append(node.sv,1.0)
                tmpsv[self.tokmap_np[words[i]]] = 0.0
                node.sv = tmpsv[:-1]
                nodes.put( (-node.eval(),node) )
            # increase qsize
            qsize += len(words)-1
        # if no finished nodes, choose the top scored paths
        if len(endnodes)==0:
            endnodes = [nodes.get() for n in range(self.overgen)]
        # choose nbest paths, back trace them
        utts = []
        for score,n in sorted(endnodes,key=operator.itemgetter(0)):
            utt = [n.wordid]
            while n.prevNode!=None:
                # back trace
                n = n.prevNode
                utt.append(n.wordid)
            utt = utt[::-1]
            utts.append((score,utt))
        return utts

    def sample(self,a,sv):
        # get 1 hot vector
        a0 = self._get1hot(a,self.da)
        sv0= self._get1hot(sv,self.dsv)
        # initial state
        h0,c0   = np.zeros(self.dh),np.zeros(self.dh)
        # container
        gens = []
        # to obtain topk generations
        for i in range(self.overgen):
            # starting node
            node = BeamSearchNode(h0,c0,None,1,0,1)
            node.sv = sv0
            node.a  = a0
            # put in queue
            nodes = [node]
            # start sampling
            while True:
                # check stopping criteria
                last_node = nodes[-1]
                if last_node.wordid==1 and len(nodes)>1:
                    break
                if len(nodes)>40: # undesirable long utt
                    break
                # expand for one time step
                words, probs, c, h = self._gen(last_node)
                # sampling according to probability
                o_sample = np.argmax(np.random.multinomial(1,probs,1))
                # put new node into the queue
                node = BeamSearchNode(h,c,last_node,words[o_sample],
                        last_node.logp+np.log10(probs[o_sample]),last_node.leng+1)
                node.sv = deepcopy(last_node.sv)
                node.a  = deepcopy(last_node.a)
                tmpsv   = np.append(node.sv,1.0)
                tmpsv[self.tokmap_np[words[o_sample]]] = 0.0
                node.sv = tmpsv[:-1]
                nodes.append( node )
            # obtain sentence
            gen = [n.wordid for n in nodes]
            # score the sentences
            score = -nodes[-1].eval()
            # make sure the generated sentence doesn't repeat
            if (score,gen) not in gens:
                gens.append((score,gen))
        # ranking generation according to score
        overgen = self.overgen if len(gens)>self.overgen else len(gens)
        gens = sorted(gens,key=operator.itemgetter(0))[:overgen]
        return gens 

    def _gen(self,node):
        
        # input word embedding
        wv_t = sigmoid(self.Wemb_np[node.wordid,:])
        # embed DA
        da_emb_t = tanh(np.dot(np.concatenate([node.a,node.sv],axis=0),self.Wfc_np))
        # compute ig, fg, og together and slice it
        gates_t = np.dot( np.concatenate(
            [wv_t,node.h,da_emb_t],axis=0),self.Wgate_np)
        ig  = sigmoid(gates_t[:self.dh])
        fg  = sigmoid(gates_t[self.dh:self.dh*2])
        og  = sigmoid(gates_t[self.dh*2:self.dh*3])
        cx_t= np.tanh(gates_t[self.dh*3:])
        # update lstm internal state
        c_t =   np.multiply(ig,cx_t) +\
                np.multiply(fg,node.c)
        # obtain new hiddne layer
        h_t = np.multiply(og,tanh(c_t))
        # compute output distribution target word prob
        o_t = softmax( np.dot(h_t,self.Who_np) )
        # make sure we won't sample unknown word
        o_t[0] = 0.0
        selected_words = np.argsort(o_t)[::-1][:self.beamwidth].tolist()
        # return results
        return selected_words, o_t[selected_words], c_t, h_t

    def loadConverseParams(self):
        self.Wemb_np    = self.params[0].get_value()
        self.Wgate_np   = self.params[1].get_value()
        self.Wfc_np     = self.params[2].get_value()
        self.Who_np     = self.params[3].get_value()


