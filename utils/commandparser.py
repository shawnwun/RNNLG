######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import argparse


def RNNLGOptParser():
    
    parser = argparse.ArgumentParser(\
            description='Default RNNLG opt parser.')

    parser.add_argument('-mode',  help='modes: train|test|adapt|knn|ngram')
    parser.add_argument('-config', help='config file to set.')

    
    return parser.parse_args()

