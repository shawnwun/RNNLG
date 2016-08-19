######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import sys
import os
import json

class DialogActParser(object):

    ## Assumption for the parser :
    ## 1. dacts are separated by the "|" token
    ## 2. slot value pairs are separated by the ";" token
    ## 3. special values specified in "resource/special_values.tx"
    ##    unify the special values by dictionary keys
    ## 4. it strips all "'" or """ token
    ## 5. output json format

    def __init__(self):
        fin = file('resource/special_values.txt')
        self.special_values = json.load(fin)
        fin.close()

    def parse(self,dact,keepValues=False):

        acttype = dact.split('(')[0]
        slt2vals= dact.split('(')[1].replace(')','').split(';')
        jsact = {'acttype':acttype,'s2v':[]}
        for slt2val in slt2vals:
            if slt2val=='': # no slot
                jsact['s2v'].append((None,None))
            elif '=' not in slt2val: # no value
                slt2val = slt2val.replace('_','').replace(' ','')
                jsact['s2v'].append( (slt2val.strip('\'\"'),'?') )
            else: # both slot and value exist 
                s,v = [x.strip('\'\"') for x in slt2val.split('=')]
                s = s.replace('_','').replace(' ','')
                for key,vals in self.special_values.iteritems():
                    if v in vals: # unify the special values
                        v = key
                if  not self.special_values.has_key(v) and\
                    not keepValues: # delexicalisation
                    v = '_'
                jsact['s2v'].append((s,v))
        return jsact

class DActFormatter(object):
    ## basic DAct formatter 
    ## 1. abstract class for Hard and Soft subclass
    ## 2. define the basic parser command
    def __init__(self):
        self.parser = DialogActParser() 
        self.special_values = self.parser.special_values.keys()
    def format(self,dact,keepValues=False):
        raise NotImplementedError('method format() hasn\'t been implemented')
    def parse(self,dact,keepValues=False):
        return self.parser.parse(dact,keepValues)

class SoftDActFormatter(DActFormatter):
    ## Soft DAct formatter
    ## 1. subclass of DActFormatter
    ## 2. main interface for parser/formatter
    ## 3. formatting the JSON DAct produced by DialogActParser 
    ##    into a feature format fed into the network

    def __init__(self):
        DActFormatter.__init__(self)
    
    def format(self,dact):
        jsact = super(SoftDActFormatter,self).parse(dact)
        mem = {}
        feature = []
        for sv in jsact['s2v']:
            s,v = sv
            if s==None: # no slot no value
                continue # skip it
            elif v=='?': # question case
                feature.append((s,v))
            elif v=='_': # categories
                if mem.has_key(s): # multiple feature values
                    feature.append((s,v+str(mem[s])))
                    mem[s] += 1
                else: # first occurance
                    feature.append((s,v+'1'))
                    mem[s] =  2
            elif v in self.special_values: # special values
                feature.append((s,v))
        feature = [('a',jsact['acttype'])]+sorted(feature)
        return feature

    def parse(self,dact,keepValues=False):
        return self.parser.parse(dact,keepValues)


class HardDActFormatter(DActFormatter):
    ## Hard DAct formatter
    ## 1. subclass of DActFormatter
    ## 2. main interface for parser/formatter
    ## 3. formatting the JSON DAct produced by DialogActParser 
    ##    into a feature format fed into the network
    ## 4. the output format is like 
    ##    ['A-inform', 'SV-count=VALUE', 'SV-type=VALUE']
    ##    the format used in EMNLP 2015 submission
    def __init__(self):
        DActFormatter.__init__(self)
    
    def format(self,dact):
        jsacts = super(HardDActFormatter,self).parse(dact)
        features = []
        mem = {}
        for jsact in jsacts:
            feature = []
            for sv in jsact['s2v']:
                s,v = sv
                if s==None: # no slot no value
                    feature.append('SV-NoSlot=NoValue')
                elif v=='?': # question case
                    feature.append('SV-'+s+'=PENDING')
                elif v=='_': # categories
                    if mem.has_key(s): # multiple feature values
                        feature.append('SV-'+s+'=VALUE'+str(mem[s]))
                        mem[s] += 1
                    else: # first occurance
                        feature.append('SV-'+s+'=VALUE')
                        mem[s] =  1
                elif v in self.special_values: # special values
                    feature.append('SV-'+s+'='+v)
            features.append(['A-'+jsact['acttype']]+sorted(feature))
        return features

    def parse(self,dact,keepValues=False):
        return self.parser.parse(dact,keepValues)

if __name__ == '__main__':
    #dadp = DialogActDelexicalizedParser()
    dadp = HardDActFormatter()

    print dadp.format("inform(type='restaurant';count='182';area=dont_care)")
    print dadp.format("reqmore()")
    print dadp.format("request(area)")
    print dadp.format("inform(name='fifth floor';address='hotel palomar 12 fourth street or rosie street')")
    print dadp.format("inform(name='fifth floor';address='hotel palomar 12 fourth street and rosie street')")
    print dadp.format("?select(food=dont_care;food='sea food')")
    print dadp.format("?select(food='yes';food='no')")
    print dadp.format("?select(battery rating=exceptional;battery rating=standard)")
    print dadp.format("suggest(weight range=heavy;weight range=light weight;weightrange=dontcare)")
    print dadp.format("?compare(name=satellite morpheus 36;warranty=1 year european;dimension=33.7 inch;name=tecra proteus 23;warranty=1 year international;dimension=27.4 inch)")
