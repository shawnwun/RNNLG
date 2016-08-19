import os
import sys
import numpy as np

print '##################################################################################################'
print '############################ Results average over 5 trained generators ###########################'
print '##################################################################################################'
print 'Model\tDomain\tavgBLEU\tavgERR\tstdBLEU\tstd-ERR\t# of Net'
dataset = [ 'counterfeit/r2h','counterfeit/h2r',
            'counterfeit/l2t','counterfeit/t2l',
            'counterfeit/r+h2l+t','counterfeit/l+t2r+h']
newdset = [ 'original/hotel','original/restaurant','original/tv',
            'original/laptop','union/l+t','union/r+h']
domain  = ['hotel','restaurant','tv','laptop','laptop','restaurant']

gtypes  = ['hlstm','sclstm','encdec']
seed    = [str(i) for i in range(1,6)]


mode    = 'all'
batch   = '1'
hidden  = '80'
percent = ['1','2','3','4','5','10','20','50','100']
obj     = 'ml'

for i in range(len(dataset)):
    dset = dataset[i]
    dom  = domain[i]
    nwset= newdset[i]
    print dset +'-'+ nwset
    for g in gtypes:
        for p in percent:
            bleus,errs = [],[]
            for s in seed:
                # name the model
                params = [g,dom,dset.replace('/','@'),'100',s]
                identify = '-'.join(params)
                 
                # log file
                log = 'log/batch/'+identify+'.log-adpML-'+nwset.replace('/','@')+'-'+p

                try:
                    fin = file(log)
                    lines = fin.readlines()
                    BLEU = lines[-1].split('\t')[1]
                    ERR  = lines[-1].split('\t')[2].split('/')[0].\
                            replace('%','')
                    #print '%.4f\t%.2f%%' % (float(BLEU),float(ERR))
                    bleus.append(float(BLEU))
                    errs.append(float(ERR))
                except:
                    pass
            bleus = np.array(bleus) 
            errs  = np.array(errs)
            print '%12s\t%4s\t%4s\t%.4f\t%.2f%%\t%d' % \
                (g,dom,p,np.mean(bleus),np.mean(errs),bleus.size)

