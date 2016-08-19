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

from basic  import *

class sclstm(BaseRLG):

    def __init__(self, gentype, vocab, beamwidth, overgen,
            vocab_size, hidden_size, batch_size, da_sizes):
        
        # calling superclass constructor
        BaseRLG.__init__(self, gentype, beamwidth, overgen,
                vocab_size, hidden_size, batch_size, da_sizes)
        self.dsv    = self.dfs[2]-self.dfs[1]
        self.da     = self.dfs[1]-self.dfs[0]
        self.vocab  = vocab
        # init params
        self._init_params()

    def _init_params(self):
        
        # word embedding weight matrix
        self.Wemb   = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.di,self.dh)).astype(theano.config.floatX))
        # lstm gate weight matrix
        self.Wgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*2+self.dsv,self.dh*3)).astype(theano.config.floatX))
        # for reading gate
        self.Wrgate = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*2+self.dsv,self.dsv)).\
                astype(theano.config.floatX))
        # for overriding the memory cell
        self.Wcx = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*2,self.dh)).astype(theano.config.floatX))
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
                self.Wgate, self.Wrgate, 
                self.Wcx,   self.Wfc,
                self.Who ]
    
    def setWordVec(self,word2vec):
        self.Wemb_np = self.Wemb.get_value()
        for w,v in word2vec.iteritems():
            self.Wemb_np[w,:] = v
        self.Wemb.set_value(self.Wemb_np)

    def _form1hot(self, hot_x, idx_x, cutoff_x):
        update_x = T.set_subtensor(hot_x[idx_x[:cutoff_x]],1.0)
        return update_x

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

        # recurrence
        [f,h,c,p],_ = theano.scan(fn=self._recur,
                sequences=[words[:-1,:],words[1:,:]],
                outputs_info=[sv_1hot,self.h0,self.c0,None],
                non_sequences=[a_1hot])
        # compute desired sent_logp by slicing
        cutoff_logp = collectSentLogp(p,cutoff_f[4],cutoff_b)
        # semantic alignment cost
        semcost =   T.sum(abs(f[0,:,:]-sv_1hot))+\
                    T.sum(abs(f[-1,:,:]))+\
                    T.sum(0.0001*(100.0**abs(f[:-1,:,:]-f[1:,:,:])))
        cost = -T.sum(cutoff_logp) + semcost
        return cost, cutoff_logp

    def _recur(self, w_t, y_t, sv_tm1, h_tm1, c_tm1, a):
        
        # input word embedding
        wv_t = T.nnet.sigmoid(self.Wemb[w_t,:])
        # compute ig, fg, og together and slice it
        gates_t = T.dot( T.concatenate([wv_t,h_tm1,sv_tm1],axis=1),self.Wgate)
        ig  = T.nnet.sigmoid(gates_t[:,:self.dh])
        fg  = T.nnet.sigmoid(gates_t[:,self.dh:self.dh*2])
        og  = T.nnet.sigmoid(gates_t[:,self.dh*2:self.dh*3])
        # compute reading rg
        rg  = T.nnet.sigmoid(T.dot(
            T.concatenate([wv_t,h_tm1,sv_tm1],axis=1),self.Wrgate))
        # compute proposed cell value
        cx_t= T.tanh(T.dot(T.concatenate([wv_t,h_tm1],axis=1),self.Wcx))
        # update DA 1-hot vector
        sv_t = rg*sv_tm1
        # update lstm internal state
        c_t = ig*cx_t + fg*c_tm1 + \
                T.tanh(T.dot(T.concatenate([a,sv_t],axis=1),self.Wfc))
        # obtain new hiddne layer
        h_t = og*T.tanh(c_t)
        # compute output distribution target word prob
        o_t = T.nnet.softmax( T.dot(h_t,self.Who) )
        p_t = o_t[T.arange(self.db),y_t]

        return sv_t, h_t, c_t, p_t

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
                # update score with sem cost
                n.logp -= np.sum(abs(n.sv))
                score = -n.eval()
                endnodes.append((score,n))
                # if reach maximum # of sentences required
                if len(endnodes)>=self.overgen: break
                else:                           continue
            # decode for one step using decoder 
            words, probs, sv, c, h = self._gen(n)
            # put them into a queue
            for i in range(len(words)):
                node = BeamSearchNode(h,c,n,words[i],
                        n.logp+np.log10(probs[i])-
                        np.sum(0.0001*(100.0**abs(sv-n.sv)))
                        ,n.leng+1)
                node.sv = sv
                node.a  = a
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
            nodes = [[-node.eval(),node]]
            # start sampling
            while True:
                # check stopping criteria
                last_node = nodes[-1][-1]
                if (last_node.wordid==1 and len(nodes)>1) or\
                        len(nodes)>40: # undesirable long utt
                    # update score with sem cost
                    last_node.logp -= np.sum(abs(last_node.sv))
                    score = -last_node.eval()
                    nodes[-1] = [score,last_node]
                    break
                # expand for one time step
                words, probs, sv, c, h = self._gen(last_node)
                # sampling according to probability
                o_sample = np.argmax(np.random.multinomial(1,probs,1))
                # put new node into the queue
                node = BeamSearchNode(h,c,last_node,words[o_sample],
                        last_node.logp+np.log10(probs[o_sample])-
                        np.sum(0.0001*(100.0**abs(sv-last_node.sv))),
                        last_node.leng+1)
                node.sv = sv
                node.a  = a0
                nodes.append( [-node.eval(),node] )
            # obtain sentence
            gen = [n.wordid for s,n in nodes]
            score = nodes[-1][0]
            # score the sentences
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
        # compute ig, fg, og together and slice it
        gates_t = np.dot( np.concatenate(
            [wv_t,node.h,node.sv],axis=0),self.Wgate_np)
        ig  = sigmoid(gates_t[:self.dh])
        fg  = sigmoid(gates_t[self.dh:self.dh*2])
        og  = sigmoid(gates_t[self.dh*2:self.dh*3])
        # compute reading rg
        rg  = sigmoid(np.dot(np.concatenate(
            [wv_t,node.h,node.sv],axis=0),self.Wrgate_np))
        # compute proposed cell value
        cx_t= np.tanh(np.dot(np.concatenate(
            [wv_t,node.h],axis=0),self.Wcx_np))
        # update DA 1-hot vector
        sv_t = np.multiply(rg,node.sv)
        # update lstm internal state
        c_t =   np.multiply(ig,cx_t) +\
                np.multiply(fg,node.c)+\
                tanh(np.dot(np.concatenate([node.a,sv_t],axis=0),self.Wfc_np))
        # obtain new hiddne layer
        h_t = np.multiply(og,tanh(c_t))
        # compute output distribution target word prob
        o_t = softmax( np.dot(h_t,self.Who_np) )
        # make sure we won't sample unknown word
        o_t[0] = 0.0
        selected_words = np.argsort(o_t)[::-1][:self.beamwidth].tolist()
        # return results
        return selected_words, o_t[selected_words], sv_t, c_t, h_t

    def loadConverseParams(self):
        self.Wemb_np    = self.params[0].get_value()
        self.Wgate_np   = self.params[1].get_value()
        self.Wrgate_np  = self.params[2].get_value()
        self.Wcx_np     = self.params[3].get_value()
        self.Wfc_np     = self.params[4].get_value()
        self.Who_np     = self.params[5].get_value()



