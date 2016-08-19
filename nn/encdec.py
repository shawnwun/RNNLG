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

class encdec(BaseRLG):

    def __init__(self, gentype, vocab, beamwidth, overgen,
            vocab_size, hidden_size, batch_size, da_sizes):
        
        # calling superclass constructor
        BaseRLG.__init__(self, gentype, beamwidth, overgen,
                vocab_size, hidden_size, batch_size, da_sizes)
        self.da     = self.dfs[1]-self.dfs[0]
        self.ds     = self.dfs[3]-self.dfs[2]
        self.dv     = self.dfs[4]-self.dfs[3]
        self.vocab  = vocab
        # init params
        self._init_params()

    def _init_params(self):
        
        # word embedding weight matrix
        self.Wemb   = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.di,self.dh)).astype(theano.config.floatX))
        # DA embedding
        npah = 0.3*np.random.uniform(-1.0,1.0,(self.da+1,self.dh))
        npsh = 0.3*np.random.uniform(-1.0,1.0,(self.ds+1,self.dh))
        npvh = 0.3*np.random.uniform(-1.0,1.0,(self.dv+1,self.dh))
        npah[self.da,:] = 0.0
        npsh[self.ds,:] = 0.0
        npvh[self.dv,:] = 0.0
        self.Wah    = theano.shared(npah.astype(theano.config.floatX))
        self.Wsh    = theano.shared(npsh.astype(theano.config.floatX))
        self.Wvh    = theano.shared(npvh.astype(theano.config.floatX))
        # attention weights
        self.Wha    = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*3,self.dh)).astype(theano.config.floatX))
        self.Vha    = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        # lstm gate weight matrix
        self.Wgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*3,self.dh*4)).astype(theano.config.floatX))
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
                self.Wah,   self.Wsh,   self.Wvh, 
                self.Wha,   self.Vha,
                self.Wgate,
                self.Who ]
    
    def setWordVec(self,word2vec):
        self.Wemb_np = self.Wemb.get_value()
        for w,v in word2vec.iteritems():
            self.Wemb_np[w,:] = v
        self.Wemb.set_value(self.Wemb_np)


    def _emb(self, a, s, v):
        a_emb = T.sum(self.Wah[a,:],axis=0)
        s_emb = self.Wsh[s,:]
        v_emb = self.Wvh[v,:]
        sv_emb= s_emb + v_emb
        return a_emb, sv_emb

    def unroll(self,a,s,v,words,cutoff_f,cutoff_b):
        
        # embed DA
        [a_emb,sv_emb],_= theano.scan(fn=self._emb,
                sequences=[a,s,v])
        sv_emb = sv_emb.dimshuffle(1,0,2)
        # recurrence
        [h,c,p],_ = theano.scan(fn=self._recur,
                sequences=[words[:-1,:],words[1:,:]],
                outputs_info=[self.h0,self.c0,None],
                non_sequences=[a_emb,sv_emb])
        # compute desired sent_logp by slicing
        cutoff_logp = collectSentLogp(p,cutoff_f[4],cutoff_b)
        cost = -T.sum(cutoff_logp) 
        return cost, cutoff_logp

    def _attend(self, sv_emb_x, wv_t, h_tm1):
        state_x = T.concatenate([wv_t,h_tm1,sv_emb_x],axis=1)
        score_x = T.dot(T.tanh(T.dot(state_x,self.Wha)),self.Vha)
        return score_x

    def _recur(self, w_t, y_t, h_tm1, c_tm1, a_emb, sv_emb):
        
        # input word embedding
        wv_t = T.nnet.sigmoid(self.Wemb[w_t,:])
        # attention
        b_t,_= theano.scan(fn=self._attend,
                sequences=[sv_emb],
                non_sequences=[wv_t,h_tm1])
        b_t = T.nnet.softmax(b_t.dimshuffle(1,0))
        sv_emb_t = T.tensordot(b_t,sv_emb,[[1,0],[0,1]])
        da_emb_t = T.tanh( a_emb+sv_emb_t )
        # compute ig, fg, og together and slice it
        gates_t = T.dot( T.concatenate([wv_t,h_tm1,da_emb_t],axis=1),self.Wgate)
        ig  = T.nnet.sigmoid(gates_t[:,:self.dh])
        fg  = T.nnet.sigmoid(gates_t[:,self.dh:self.dh*2])
        og  = T.nnet.sigmoid(gates_t[:,self.dh*2:self.dh*3])
        cx_t= T.tanh( gates_t[:,self.dh*3:] )
        # update lstm internal state
        c_t = ig*cx_t + fg*c_tm1
        # obtain new hiddne layer
        h_t = og*T.tanh(c_t)
        # compute output distribution target word prob
        o_t = T.nnet.softmax( T.dot(h_t,self.Who) )
        p_t = o_t[T.arange(self.db),y_t]

        return h_t, c_t, p_t

    def _npemb(self,a,s,v):
        return  np.sum(self.Wah_np[a,:],axis=0),\
                self.Wsh_np[s,:]+self.Wvh_np[v,:]

    def beamSearch(self,a,s,v):
        # embed DA
        a_emb, sv_emb = self._npemb(a,s,v)
        # end nodes
        endnodes = []
        # initial layers
        h0,c0 = np.zeros(self.dh),np.zeros(self.dh)
        # starting node
        node = BeamSearchNode(h0,c0,None,1,0,1)
        node.sv = sv_emb
        node.a  = a_emb
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
                node.sv = sv_emb
                node.a  = a_emb
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

    def sample(self,a,s,v):
        # embed DA
        a_emb, sv_emb = self._npemb(a,s,v)
        # initial state
        h0,c0   = np.zeros(self.dh),np.zeros(self.dh)
        # container
        gens = []
        # to obtain topk generations
        for i in range(self.overgen):
            # starting node
            node = BeamSearchNode(h0,c0,None,1,0,1)
            node.sv = sv_emb
            node.a  = a_emb
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
                node.sv = sv_emb
                node.a  = a_emb
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
        # attention
        b_t = np.zeros((node.sv.shape[0]))
        for j in range(node.sv.shape[0]):
            b_t[j] = np.dot(tanh(np.dot(
                np.concatenate([wv_t,node.h,node.sv[j]],axis=0),
                self.Wha_np)),self.Vha_np)
        b_t = softmax(b_t)
        sv_emb_t = np.dot(b_t,node.sv)
        da_emb_t = tanh( node.a+sv_emb_t )
        # compute ig, fg, og together and slice it
        gates_t = np.dot( np.concatenate([wv_t,node.h,da_emb_t],axis=0),
                self.Wgate_np)
        ig  = sigmoid(gates_t[:self.dh])
        fg  = sigmoid(gates_t[self.dh:self.dh*2])
        og  = sigmoid(gates_t[self.dh*2:self.dh*3])
        cx_t= tanh( gates_t[self.dh*3:] )
        # update lstm internal state
        c_t = np.multiply(ig,cx_t) + np.multiply(fg,node.c)
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
        self.Wah_np     = self.params[1].get_value()
        self.Wsh_np     = self.params[2].get_value()
        self.Wvh_np     = self.params[3].get_value()
        self.Wha_np     = self.params[4].get_value()
        self.Vha_np     = self.params[5].get_value()
        self.Wgate_np   = self.params[6].get_value()
        self.Who_np     = self.params[7].get_value()


