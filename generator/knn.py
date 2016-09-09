######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
from __future__ import print_function

from math import sqrt

from copy import deepcopy

from loader.data_reader import *
from loader.generation_scorer import *

try:

    from ConfigParser import SafeConfigParser
except ImportError:

    from configparser import SafeConfigParser


class KNN(object):
    def __init__(self, config=None, opts=None):
        # not enough info to execute
        if config == None and opts == None:
            print("Please specify command option or config file ...")
            return
        # config parser
        parser = SafeConfigParser()
        parser.read(config)

        self.debug = parser.getboolean('knn', 'debug')
        self.seed = parser.getint('knn', 'random_seed')
        self.obj = 'dt'
        self.trainfile = parser.get('knn', 'train')
        self.validfile = parser.get('knn', 'valid')
        self.testfile = parser.get('knn', 'test')
        self.vocabfile = parser.get('knn', 'vocab')
        self.domain = parser.get('knn', 'domain')
        self.percentage = float(parser.getfloat('knn', 'percentage')) / 100.0
        # Setting generation specific parameters
        self.topk = parser.getint('knn', 'topk')
        self.detectpairs = parser.get('knn', 'detectpairs')
        self.verbose = parser.getint('knn', 'verbose')
        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        np.set_printoptions(precision=4)
        # setting data reader, processors, and lexicon
        self.setup_side_operators()

    def test_knn(self):

        ######## train KNN generator by grouping ########
        da2sents = {}
        templates = self.reader.read_all(mode='train') + \
                    self.reader.read_all(mode='valid')
        for a, sv, s, v, sents, dact, base in templates:
            key = (tuple(a), tuple(sv))
            if da2sents.has_key(key):
                da2sents[key].extend(sents)
                da2sents[key] = list(set(da2sents[key]))
            else:
                da2sents[key] = sents

        ######## test KNN generator on test set ######### 
        if self.debug:
            print('start KNN generation ...')

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
            # score DA similarity between testing example and train+valid set
            template_ranks = []
            for da_t, sents_t in iteritems(da2sents):
                a_t, sv_t = [set(x) for x in da_t]
                score = float(len(a_t.intersection(set(a))) + \
                              len(sv_t.intersection(set(sv)))) / \
                        sqrt(len(a_t) + len(sv_t)) / sqrt(len(a) + len(sv))
                template_ranks.append([score, sents_t])
            # rank templates
            template_ranks = sorted(template_ranks, key=operator.itemgetter(0))
            gens = deepcopy(template_ranks[-1][1])
            score = template_ranks[-1][0]
            random.shuffle(gens)
            gens = gens[:self.topk] if len(gens) > self.topk else gens
            # for slot error rate scoring
            felements = [self.reader.cardinality[x + self.reader.dfs[1]] \
                         for x in sv]
            # print(results)
            print(dact)
            print('Sim\tTSER\tASER\tGen')
            for i in range(len(gens)):
                # score slot error rate
                cnt, total, caty = self.gentscorer.score_error(a, felements,
                                                               self.reader.delexicalise(gens[i], dact))
                gens[i] = self.reader.lexicalise(gens[i], dact)
                # accumulate slot error cnts
                gencnts[0] += cnt
                gencnts[1] += total
                gencnts[2] += caty
                print('%.4f\t%d\t%d\t%s' % (score, total, caty, gens[i]))
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
            parallel_corpus.append([[g for g in gens], sents])
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

    def setup_side_operators(self):
        # initialise data reader
        self.reader = DataReader(self.seed, self.domain, self.obj,
                                 self.vocabfile, self.trainfile, self.validfile, self.testfile,
                                 self.percentage, self.verbose, lex_cutoff=4)
        # setting generation scorer
        self.gentscorer = GenerationScorer(self.detectpairs)
