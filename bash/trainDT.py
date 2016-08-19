import os
import sys
from configTemplate import template
import os.path
from hpc_submit import *

dataset = [ 'counterfeit/l2t','counterfeit/t2l',
            'counterfeit/r2h','counterfeit/h2r',
            'counterfeit/r+h2l+t','counterfeit/l+t2r+h']
domain  = ['tv','laptop','hotel','restaurant','laptop','restaurant']
newdset = [ 'original/tv','original/laptop','original/hotel',
            'original/restaurant','union/l+t','union/r+h']

gtypes  = ['hlstm','sclstm','encdec']
seed    = [str(i) for i in range(1,6)]

mode    = 'all'
batch   = '50'
hidden  = '80'
percent = ['1','2','3','4','5','10','20','50','100']
obj     = 'dt'
vecfile = 'vec/vectors-80.txt'
dec     = 'beam'

for i in range(len(dataset)):
    dset = dataset[i]
    nwset= newdset[i]
    dom  = domain[i]
    for p in percent:
        for g in gtypes:
            for s in seed:
                # name the model
                params = [g,dom,dset.replace('/','@'),'100',s]
                identify = '-'.join(params)
                oldmodel = 'model/batch/'+identify+'.model-adpML-'+nwset.replace('/','@')+'-'+p
                newmodel = 'model/batch/'+identify+'.model-adpDT-'+nwset.replace('/','@')+'-'+p
                                    
                # writing config file
                text = template(newmodel,mode,obj,batch,g,hidden,
                        dom,nwset,p,s,vecfile=vecfile,decode=dec,
                        lr='0.05',overgen='50')
                config ='config/batch/'+identify+'.cfg-adpDT-'+nwset.replace('/','@')+'-'+p

                fout = file(config,'w')
                fout.write(text)
                fout.close()

                # script and log file
                scp = 'scp/batch/'+identify+'.sh-adpDT-'+nwset.replace('/','@')+'-'+p
                log = 'log/batch/'+identify+'.log-adpDT-'+nwset.replace('/','@')+'-'+p
                slurm   = 'scp/batch/slurm-'+identify+'.darwin'

                # writing script file
                fout = file(scp,'w')
                fout.write('#!/bin/sh\n')
                fout.write('#$ -S /bin/bash\n')
                fout.write('cp '+oldmodel+' '+newmodel+'\n')
                fout.write('rm '+log+'\n')
                fout.write('python main.py -config '+config+\
                        ' -mode adapt\n')
                fout.write('python main.py -config '+config+\
                        ' -mode test > '+log+'\n')
                fout.close()
                os.chmod(scp,0755)

                # create slurm script to submit jobs
                fout = file(slurm,'w')
                fout.write(slurm_scp('gen',scp))
                fout.close()

                # submit jobs
                os.system('sbatch '+slurm)

                #os.system('sh '+scp)
                #raw_input()
