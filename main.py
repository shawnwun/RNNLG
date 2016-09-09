######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################

import warnings

from generator.knn import KNN
from generator.net import Model
from generator.ngram import Ngram
from utils.commandparser import RNNLGOptParser

warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == '__main__':

    args = RNNLGOptParser()
    config = args.config

    if args.mode == 'knn':
        knn = KNN(config, args)
        knn.test_knn()
    elif args.mode == 'ngram':
        ngram = Ngram(config, args)
        ngram.test_ngram()
    # Otherwise, we run the NN case
    else:
        model = Model(config, args)
        if args.mode == 'train' or args.mode == 'adapt':
            model.train_net()
        elif args.mode == 'test':
            model.test_net()

# not supported yet
"""
elif args.mode=='realtime':
    while True:
        dact=raw_input('Target dialogue act: ')
        sents, errs = model.genSent(dact)
        for s in sents:
            print(s)
        print()
"""
