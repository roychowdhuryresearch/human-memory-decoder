import numpy as np

SF = 2000

SPIKE_REGION = {"p1": ['HPC', 'PHC', 'PHC', 'PARS', 'CC', 'LTC'], 
                "p2": ['HPC', 'HPC', 'PHC', 'ERC', 'FC', 'FC', 'CC', 'VIS'], 
                'p3': ['HPC', 'HPC', 'AMY', 'FC', 'FC', 'PARS', 'CC', 'FUS', 'LTC'], 
                'p4': ['HPC', 'HPC', 'AMY', 'PHC', 'ERC'], 
                'p5': ['HPC', 'HPC', 'AMY', 'AMY', 'FC', 'FC', 'FC', 'FC', 'CC', 'CC'], 
                'p6': ['HPC', 'HPC', 'AMY', 'AMY', 'PHC', 'ERC', 'FC', 'CC', 'INS', 'INS'], 
                'p7': ['HPC', 'HPC', 'AMY', 'AMY', 'ERC', 'ERC', 'PARS', 'CC', 'CC'], 
                'p8': ['HPC', 'PHC', 'FC', 'FC', 'INS'], 
                'p9': ['AMY', 'ERC', 'ERC', 'FC', 'CC', 'CC', 'MCC', 'FUS', 'INS', 'INS', 'INS', 'INS'], 
                'p10': ['HPC', 'HPC', 'HPC', 'AMY', 'ERC', 'FC', 'FC', 'FC', 'FUS', 'INS', 'LTC', 'VIS'],
                }

SPIKE_CHANNEL = {'p1':None, 'p2':None, 'p3':None, 'p4':None, 'p5':None, 'p6':None, 'p7':None, 'p8':None, 'p9':None, 'p10':None}
SPIKE_FRAME = {'p1':50, 'p2':50, 'p3':50, 'p4':50, 'p5':50, 'p6':50, 'p7':50, 'p8':50, 'p9':50, 'p10':50}
LFP_CHANNEL = {'p1':None, 'p2':None, 'p3':None, 'p4':None, 'p5':None, 'p6':None, 'p7':None, 'p8':None, 'p9':None, 'p10':None}
LFP_FRAME = {'p1':500, 'p2':500, 'p3':500, 'p4':500, 'p5':500, 'p6':500, 'p7':500, 'p8':500, 'p9':500, 'p10':500}

LABELS8 = ['WhiteHouse', 'CIA', 'Sacrifice', 'Handcuff', 'J.Bauer', 'B.Buchanan', 'A.Fayed', 'A.Amar']
