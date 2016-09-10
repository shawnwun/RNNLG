######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
from __future__ import print_function

import os
import pickle as pk
import time
from math import log10

from loader.data_reader import *
from loader.generation_scorer import *
from nn.nn_generator import *

try:

    from ConfigParser import SafeConfigParser
except ImportError:

    from configparser import SafeConfigParser

# theano debugging flags
"""
theano.config.compute_test_value = 'warn'
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'
theano.gof.compilelock.set_lock_status(False)
"""


class Model(object):
    #######################################################################
    # all variables that needs to be save and load from model file, indexed 
    # by their names
    #######################################################################
    params_vars = ['self.params_np']
    learn_vars = ['self.lr', 'self.lr_decay', 'self.beta', 'self.seed',
                  'self.min_impr', 'self.llogp', 'self.debug', 'self.valid_logp',
                  'self.lr_divide']
    mode_vars = ['self.mode', 'self.obj', 'self.gamma', 'self.batch']
    data_vars = ['self.domain', 'self.wvecfile', 'self.modelfile',
                 'self.vocabfile', 'self.trainfile', 'self.validfile', 'self.testfile',
                 'self.percentage']
    gen_vars = ['self.topk', 'self.beamwidth', 'self.overgen',
                'self.detectpairs', 'self.verbose']
    model_vars = ['self.gentype', 'self.di', 'self.dh']

    #################################################################
    ################### Initialisation ##############################
    #################################################################
    def __init__(self, config=None, opts=None):
        # not enough info to execute
        if config == None and opts == None:
            print("Please specify command option or config file ...")
            return
        # config parser
        parser = SafeConfigParser()
        parser.read(config)
        # loading pretrained model if any
        self.modelfile = parser.get('data', 'model')
        if opts:    self.mode = opts.mode
        # check model file exists or not 
        if os.path.isfile(self.modelfile):
            if not opts:
                self.load_net(parser, None)
            else:
                self.load_net(parser, opts.mode)
        # Otherwise, initialize a new model
        else:
            self.init_net(config, opts)
            self.update_numpy_params()

    def init_net(self, config, opts=None):

        print('\n\ninit net from scrach ... ')

        # config parser
        parser = SafeConfigParser()
        parser.read(config)

        # setting learning hyperparameters 
        self.debug = parser.getboolean('learn', 'debug')
        if self.debug:
            print('loading settings from config file ...')
        self.seed = parser.getint('learn', 'random_seed')
        self.lr_divide = parser.getint('learn', 'lr_divide')
        self.lr = parser.getfloat('learn', 'lr')
        self.lr_decay = parser.getfloat('learn', 'lr_decay')
        self.beta = parser.getfloat('learn', 'beta')
        self.min_impr = parser.getfloat('learn', 'min_impr')
        self.llogp = parser.getfloat('learn', 'llogp')
        # setting training mode
        self.mode = parser.get('train_mode', 'mode')
        self.obj = parser.get('train_mode', 'obj')
        self.gamma = parser.getfloat('train_mode', 'gamma')
        self.batch = parser.getint('train_mode', 'batch')
        # setting file paths
        if self.debug:
            print('loading file path from config file ...')
        self.wvecfile = parser.get('data', 'wvec')
        self.trainfile = parser.get('data', 'train')
        self.validfile = parser.get('data', 'valid')
        self.testfile = parser.get('data', 'test')
        self.vocabfile = parser.get('data', 'vocab')
        self.domain = parser.get('data', 'domain')
        self.percentage = float(parser.getfloat('data', 'percentage')) / 100.0
        # Setting generation specific parameters
        self.topk = parser.getint('gen', 'topk')
        self.overgen = parser.getint('gen', 'overgen')
        self.beamwidth = parser.getint('gen', 'beamwidth')
        self.detectpairs = parser.get('gen', 'detectpairs')
        self.verbose = parser.getint('gen', 'verbose')
        self.decode = parser.get('gen', 'decode')
        # setting rnn configuration
        self.gentype = parser.get('generator', 'type')
        self.dh = parser.getint('generator', 'hidden')
        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        np.set_printoptions(precision=4)
        # setting data reader, processors, and lexicon
        self.setup_delegates()
        # network size
        self.di = len(self.reader.vocab)
        # logp for validation set
        self.valid_logp = 0.0
        # start setting networks 
        self.init_model()
        self.model.config_theano()

    def init_model(self):
        #################################################################
        #################### Model Initialisation #######################
        #################################################################
        if self.debug:
            print('setting network structures using theano variables ...')
        ###########################################################
        ############## Setting Recurrent Generator ################
        ###########################################################
        if self.debug:
            print('\tsetting recurrent generator, type: %s ...' % self.gentype)
        self.model = NNGenerator(self.gentype, self.reader.vocab,
                                 self.beamwidth, self.overgen,
                                 self.di, self.dh, self.batch, self.reader.dfs,
                                 self.obj, self.mode, self.decode,
                                 self.reader.token_map_to_indices())
        # setting word vectors
        if self.wvecfile != 'None':
            self.model.set_word_vec(self.reader.read_vec_file(
                self.wvecfile, self.reader.vocab))
        if self.debug:
            print('\t\tnumber of parameters : %8d' % self.model.num_params())
            print('\tthis may take up to several minutes ...')

    #################################################################
    ####################### Training ################################
    #################################################################
    def train_net(self):
        if self.obj == 'ml':
            self.train_net_ml()
        elif self.obj == 'dt':
            self.train_net_dt()

    def train_net_ml(self):
        ######## training RNN generator with early stopping ######### 
        if self.debug:
            print('start network training ...')
        epoch = 0
        lr_divide = 0
        llr_divide = -1

        while True:
            # training phase
            epoch += 1
            tic = time.time()
            wcn, num_sent, train_logp = 0.0, 0.0, 0.0
            while True:
                # read data point
                data = self.reader.read(mode='train', batch=self.batch)
                if data == None:
                    break
                # set regularization , once per ten times
                reg = 0 if random.randint(0, 9) == 5 else self.beta
                # unfold data point
                a, sv, s, v, words, _, _, cutoff_b, cutoff_f = data
                # train net using current example 
                train_logp += self.model.train(a, sv, s, v, words,
                                               cutoff_f, cutoff_b, self.lr, reg)
                # count words and sents 
                wcn += np.sum(cutoff_f - 1)
                num_sent += cutoff_b
                # log message 
                if self.debug and num_sent % 100 == 0:
                    print('Finishing %8d sent in epoch %3d\r' % (num_sent, epoch), end="")
                    sys.stdout.flush()
            # log message
            sec = (time.time() - tic) / 60.0
            if self.debug:
                print('Epoch %3d, Alpha %.6f, TRAIN entropy:%.2f, Time:%.2f mins,' % \
                      (epoch, self.lr, -train_logp / log10(2) / wcn, sec), end="")
                sys.stdout.flush()

            # validation phase
            self.valid_logp, wcn = 0.0, 0.0
            while True:
                # read data point
                data = self.reader.read(mode='valid', batch=self.batch)
                if data == None:
                    break
                # unfold data point
                a, sv, s, v, words, _, _, cutoff_b, cutoff_f = data
                # validating
                self.valid_logp += self.model.test(a, sv, s, v, words,
                                                   cutoff_f, cutoff_b)
                wcn += np.sum(cutoff_f - 1)
            # log message
            if self.debug:
                print('VALID entropy:%.2f' % -(self.valid_logp / log10(2) / wcn))

            # decide to throw/keep weights
            if self.valid_logp < self.llogp:
                self.update_theano_params()
            else:
                self.update_numpy_params()
            self.save_net()
            # learning rate decay
            if lr_divide >= self.lr_divide:
                self.lr *= self.lr_decay
            # early stopping
            if self.valid_logp * self.min_impr < self.llogp:
                if lr_divide < self.lr_divide:
                    self.lr *= self.lr_decay
                    lr_divide += 1
                else:
                    self.save_net()
                    print('Training completed.')
                    break
            # set last epoch objective value
            self.llogp = self.valid_logp

    # Discriminative Training / Expected Objective Training
    def train_net_dt(self):
        # start 
        if self.debug:
            print('start network training with expected objective ...')

        # examples
        train_examples = self.reader.read_all(mode='train')
        valid_examples = self.reader.read_all(mode='valid')

        ######## training with early stopping ######### 
        epoch = 0
        lr_divide = 1
        self.lobj = 100000000.0
        while True:
            # training phase
            tic = time.time()
            epoch += 1
            train_obj = 0.0
            train_bleu = 0.0
            train_err = 0.0
            num_sent = 0.0

            # load generation parameters
            self.model.load_converse_params()

            for example in train_examples:
                # fetch one example
                a, sv, s, v, sents, dact, bases = example
                # generate sentences
                gens = self.model.gen(a, sv, s, v)
                # for slot error rate scoring
                felements = [self.reader.cardinality[x + self.reader.dfs[1]] \
                             for x in sv]

                # post processing and generate training data
                wordids, bleus, errors, lengs = [], [], [], []
                for i in range(len(gens)):
                    penalty, gen = gens[i]
                    # replace word id with actual words
                    words = ' '.join([self.reader.vocab[x] for x in gen[1:-1]])
                    # score slot error rate
                    cnt, total, caty = self.gentscorer.score_error(a, felements, words)
                    # compute sentence bleu
                    parallel_sents = [[[words], sents]]
                    sbleu = self.gentscorer.score_sbleu(parallel_sents)
                    # containers
                    wordids.append(gen)
                    bleus.append(sbleu)
                    errors.append(total)
                    lengs.append([len(a), len(sv), len(s), len(v), len(gen)])
                # padding samples to the same length
                maxbatch = len(wordids)
                maxleng = max([x[-1] for x in lengs])
                for i in range(len(wordids)):
                    wordids[i].extend([1] * (maxleng - lengs[i][-1]))
                # padding samples to have the same batch
                lengs.extend([[0.0 for x in lengs[0]]
                              for x in range(self.batch - len(lengs))])
                wordids.extend([deepcopy(wordids[0])
                                for x in range(self.batch - len(wordids))])
                bleus.extend([0.0 for x in range(self.batch - len(wordids))])
                errors.extend([0.0 for x in range(self.batch - len(wordids))])
                # swap indexes
                lengs = np.swapaxes(np.array(lengs), 0, 1)
                wordids = np.swapaxes(np.array(wordids), 0, 1)

                # DT training
                reg = 0 if random.randint(0, 9) == 5 else self.beta
                xObj, xBLEU, xERR, senp = self.model.trainObj(
                    [a] * self.batch, [sv] * self.batch,
                    [s] * self.batch, [v] * self.batch,
                    wordids, lengs, maxbatch, bleus, errors, self.gamma,
                    self.lr, reg)

                # update generator parameters
                self.model.load_converse_params()
                # accumulate statistics
                train_bleu += xBLEU
                train_err += xERR
                train_obj += xObj
                num_sent += 1
                if self.debug and num_sent % 1 == 0:
                    print('Finishing %8d sent in epoch %3d\r' % \
                          (num_sent, epoch), end="")
                    sys.stdout.flush()
            sec = (time.time() - tic) / 60.0
            if self.debug:
                print(
                    'Epoch %2d, Alpha %.4f, TRAIN Obj:%.4f, Expected BLEU:%.4f, Expected ERR:%.4f, Time:%.2f mins,' % (
                    epoch, self.lr, train_obj / float(num_sent), train_bleu / float(num_sent),
                    train_err / float(num_sent), sec), end="")
                sys.stdout.flush()

            # validation phase
            self.valid_obj = 0.0
            num_sent = 0.0
            for example in valid_examples:
                # fetch one example
                a, sv, s, v, sents, dact, bases = example
                # generate sentences
                gens = self.model.gen(a, sv, s, v)
                # for slot error rate scoring
                felements = [self.reader.cardinality[x + self.reader.dfs[1]] \
                             for x in sv]

                # post processing and generate training data
                wordids, bleus, errors, lengs = [], [], [], []
                for i in range(len(gens)):
                    penalty, gen = gens[i]
                    # replace word id with actual words
                    words = ' '.join([self.reader.vocab[x] for x in gen[1:-1]])
                    # score slot error rate
                    cnt, total, caty = self.gentscorer.score_error(a, felements, words)
                    # compute sentence bleu
                    parallel_sents = [[[words], sents]]
                    sbleu = self.gentscorer.score_sbleu(parallel_sents)
                    # containers
                    wordids.append(gen)
                    bleus.append(sbleu)
                    errors.append(total)
                    lengs.append([len(a), len(sv), len(s), len(v), len(gen)])
                # padding samples to the same length
                maxbatch = len(wordids)
                maxleng = max([x[-1] for x in lengs])
                for i in range(len(wordids)):
                    wordids[i].extend([1] * (maxleng - lengs[i][-1]))
                # padding samples to have the same batch
                lengs.extend([[0.0 for x in lengs[0]]
                              for x in range(self.batch - len(lengs))])
                wordids.extend([deepcopy(wordids[0])
                                for x in range(self.batch - len(wordids))])
                bleus.extend([0.0 for x in range(self.batch - len(wordids))])
                errors.extend([0.0 for x in range(self.batch - len(wordids))])
                # swap indexes
                lengs = np.swapaxes(np.array(lengs), 0, 1)
                wordids = np.swapaxes(np.array(wordids), 0, 1)

                # DT validation
                reg = 0 if random.randint(0, 9) == 5 else self.beta
                xObj, xBLEU, xERR = self.model.testObj(
                    [a] * self.batch, [sv] * self.batch,
                    [s] * self.batch, [v] * self.batch,
                    wordids, lengs, maxbatch, bleus, errors, self.gamma)
                self.valid_obj += xObj
                num_sent += 1

            if self.debug:
                print('VALID Obj:%.3f' % (self.valid_obj / float(num_sent)))

            # decide to throw/keep weights
            if self.valid_obj > self.lobj:  # throw weight
                self.update_theano_params()
            else:  # keep weight
                self.update_numpy_params()

            # early stopping
            if self.valid_obj > self.lobj:
                if lr_divide < self.lr_divide:
                    lr_divide += 1
                else:
                    self.save_net()
                    print('Training completed.')
                    break

            if self.valid_obj < self.lobj:
                self.lobj = self.valid_obj

            if lr_divide >= self.lr_divide:
                self.lr *= self.lr_decay


    #################################################################
    ####################### Generation ##############################
    #################################################################
    def test_net(self):
        ######## test RNN generator on test set ######### 
        if self.debug:
            print('start network testing ...')
        self.model.load_converse_params()

        # container
        parallel_corpus, hdc_corpus = [], []
        # slot error counts
        gencnts, refcnts = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        while True:
            # read data point
            data = self.reader.read(mode='test', batch=1)
            if data == None:
                break
            a, sv, s, v, sents, dact, bases, cutoff_b, cutoff_f = data
            # remove batch dimension
            a, sv, s, v = a[0], sv[0], s[0], v[0]
            sents, dact, bases = sents[0], dact[0], bases[0]
            # generate sentences
            gens = self.model.gen(a, sv, s, v)
            # for slot error rate scoring
            felements = [self.reader.cardinality[x + self.reader.dfs[1]] \
                         for x in sv]

            # post processing
            for i in range(len(gens)):
                penalty, gen = gens[i]
                # replace word id with actual words
                gen = ' '.join([self.reader.vocab[x] for x in gen[1:-1]])
                # score slot error rate
                cnt, total, caty = self.gentscorer.score_error(a, felements, gen)
                # update score by categorical slot errors
                penalty += caty
                # lexicalise back
                gens[i] = (penalty, self.reader.lexicalise(gen, dact))
            # get the top-k for evaluation
            gens = sorted(gens, key=operator.itemgetter(0))[:self.topk]
            # print(results)
            print(dact)
            print('Penalty\tTSER\tASER\tGen')
            for penalty, gen in gens:
                # score slot error rate
                cnt, total, caty = self.gentscorer.score_error(a, felements,
                                                               self.reader.delexicalise(gen, dact))
                # accumulate slot error cnts
                gencnts[0] += cnt
                gencnts[1] += total
                gencnts[2] += caty
                print('%.4f\t%d\t%d\t%s' % (penalty, total, caty, gen))
            print('\n')

            # compute gold standard slot error rate
            for sent in sents:
                # score slot error rate
                cnt, total, caty = self.gentscorer.score_error(a, felements,
                                                               self.reader.delexicalise(sent, dact))
                # accumulate slot error cnts
                refcnts[0] += cnt
                refcnts[1] += total
                refcnts[2] += caty

            # accumulate score for bleu score computation         
            parallel_corpus.append([[g for p, g in gens], sents])
            hdc_corpus.append([bases[:1], sents])

        bleuModel = self.gentscorer.score_bleu(parallel_corpus)
        bleuHDC = self.gentscorer.score_bleu(hdc_corpus)
        print('##############################################')
        print('BLEU SCORE & SLOT ERROR on GENERATED SENTENCES')
        print('##############################################')
        print('Metric       :\tBLEU\tT.ERR\tA.ERR')
        print('HDC          :\t%.4f\t%2.2f%%\t%2.2f%%' % (bleuHDC, 0.0, 0.0))
        print('Ref          :\t%.4f\t%2.2f%%\t%2.2f%%' % (1.0,
                                                          100 * refcnts[1] / refcnts[0], 100 * refcnts[2] / refcnts[0]))
        print('----------------------------------------------')
        print('This Model   :\t%.4f\t%2.2f%%\t%2.2f%%' % (bleuModel,
                                                          100 * gencnts[1] / gencnts[0], 100 * gencnts[2] / gencnts[0]))

    #################################################################
    #################### Utility Functions ##########################
    #################################################################
    def update_theano_params(self):
        # update theano from np
        self.model.set_params(self.params_np)

    def update_numpy_params(self):
        # update np from theano
        self.params_np = self.model.get_params()

    def save_net(self):
        if self.debug:
            print('saving net to file ... ')
        self.update_numpy_params()
        locals_ = locals()
        bundle = {
            'learn': dict([(name, eval(name, globals(), locals_)) for name in self.learn_vars]),
            'data': dict([(name, eval(name, globals(), locals_)) for name in self.data_vars]),
            'gen': dict([(name, eval(name, globals(), locals_)) for name in self.gen_vars]),
            'model': dict([(name, eval(name, globals(), locals_)) for name in self.model_vars]),
            'mode': dict([(name, eval(name, globals(), locals_)) for name in self.mode_vars]),
            'params': dict([(name, eval(name, globals(), locals_)) for name in self.params_vars])
        }
        pk.dump(bundle, open(self.modelfile, 'wb'))

    def load_net(self, parser, mode):

        print('\n\nloading net from file %s ... ' % self.modelfile)
        bundle = pk.load(open(self.modelfile, 'rb'))
        # load learning variables from model
        # if adaptation, load from config file
        if mode == 'adapt':
            self.lr = parser.getfloat('learn', 'lr')
            self.llogp = parser.getfloat('learn', 'llogp')
            self.valid_logp = 0.0
        else:
            self.lr = bundle['learn']['self.lr']
            self.llogp = bundle['learn']['self.llogp']
            self.valid_logp = bundle['learn']['self.valid_logp']

        self.lr_decay = bundle['learn']['self.lr_decay']
        self.lr_divide = bundle['learn']['self.lr_divide']
        self.beta = bundle['learn']['self.beta']
        self.seed = bundle['learn']['self.seed']
        self.min_impr = bundle['learn']['self.min_impr']
        self.debug = bundle['learn']['self.debug']
        # network parameters 
        self.params_np = bundle['params']['self.params_np']
        # always load train/valid/test file from config
        self.domain = parser.get('data', 'domain')
        self.trainfile = parser.get('data', 'train')
        self.validfile = parser.get('data', 'valid')
        self.testfile = parser.get('data', 'test')
        self.percentage = float(parser.getfloat('data', 'percentage')) / 100.0
        # load other data files from model
        self.vocabfile = bundle['data']['self.vocabfile']
        self.wvecfile = bundle['data']['self.wvecfile']
        # load model name from config 
        self.modelfile = parser.get('data', 'model')
        # Note: always load generation variables from config
        self.topk = parser.getint('gen', 'topk')
        self.overgen = parser.getint('gen', 'overgen')
        self.beamwidth = parser.getint('gen', 'beamwidth')
        self.detectpairs = parser.get('gen', 'detectpairs')
        self.decode = parser.get('gen', 'decode')
        self.verbose = parser.getint('gen', 'verbose')
        # load model architectures from model
        self.gentype = bundle['model']['self.gentype']
        self.di = bundle['model']['self.di']
        self.dh = bundle['model']['self.dh']
        # load training mode from config
        self.mode = parser.get('train_mode', 'mode')
        self.obj = parser.get('train_mode', 'obj')
        self.gamma = parser.getfloat('train_mode', 'gamma')
        self.batch = parser.getint('train_mode', 'batch')

        self.setup_delegates()
        self.init_model()
        self.update_theano_params()

        if mode == 'train' or mode == 'adapt':
            self.model.config_theano()

    def setup_delegates(self):
        # initialise data reader
        self.reader = DataReader(self.seed, self.domain, self.obj,
                                 self.vocabfile, self.trainfile, self.validfile, self.testfile,
                                 self.percentage, self.verbose, lex_cutoff=4)
        # setting generation scorer
        self.gentscorer = GenerationScorer(self.detectpairs)
