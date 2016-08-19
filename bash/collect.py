import os
import sys
import numpy as np

print '##################################################################################################'
print '############################ Results average over 5 trained generators ###########################'
print '##################################################################################################'
print 'Model\tDomain\tavgBLEU\tavgERR\tstdBLEU\tstd-ERR\t# of Net'

dataset = ['sfxrestaurant','sfxhotel','laptop','tv']
domain  = ['restaurant','hotel','laptop','tv']
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
    print dom
    for g in gtypes:
        for p in percent:
            bleus,errs = [],[]
            for s in seed:
                # name the model
                params = [g,dom,p,s]
                identify = '-'.join(params)
                model = 'model/batch/'+identify+'.model'
                 
                # log file
                log = 'log/batch/'+identify+'.log'

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

