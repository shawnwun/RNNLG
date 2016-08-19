import os
import sys
from configTemplate import template
import os.path
from hpc_submit import *
"""
dataset = [ 'original/laptop','original/tv',\
            'original/hotel','original/restaurant']
domain  = ['laptop','tv','hotel','restaurant']
"""
dataset = [ 'counterfeit/l2t','counterfeit/t2l',
            'counterfeit/r2h','counterfeit/h2r',
            'counterfeit/r+h2l+t','counterfeit/l+t2r+h']
domain  = ['tv','laptop','hotel','restaurant','laptop','restaurant']

gtypes  = ['hlstm','sclstm','encdec']
seed    = [str(i) for i in range(1,6)]

mode    = 'all'
batch   = '1'
hidden  = '80'
percent = ['100']#['1','2','3','4','5','10','20','50','100']
obj     = 'ml'
vecfile = 'vec/vectors-80.txt'
dec     = 'beam'

for i in range(len(dataset)):
    dset = dataset[i]
    dom  = domain[i]
    for p in percent:
        for g in gtypes:
            for s in seed:
                # name the model
                params = [g,dom,dset.replace('/','@'),p,s]
                identify = '-'.join(params)
                model = 'model/batch/'+identify+'.model'
                                    
                # writing config file
                text = template(model,mode,obj,batch,g,hidden,
                        dom,dset,p,s,vecfile=vecfile,decode=dec) 
                config ='config/batch/'+identify+'.cfg'                    
                fout = file(config,'w')
                fout.write(text)
                fout.close()

                # script and log file
                scp = 'scp/batch/'+identify+'.sh'
                log = 'log/batch/'+identify+'.log'
                slurm   = 'scp/batch/slurm-'+identify+'.darwin'

                # writing script file
                fout = file(scp,'w')
                fout.write('#!/bin/sh\n')
                fout.write('#$ -S /bin/bash\n')
                fout.write('rm '+model+'\n')
                fout.write('rm '+log+'\n')
                fout.write('python main.py -config '+config+\
                        ' -mode train\n')
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
