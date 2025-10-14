import numpy as np

SF = 2000

SPIKE_REGION = {"562": ['HPC', 'PHC', 'PHC', 'PARS', 'CC', 'LTC'], 
                "563": ['HPC', 'HPC', 'PHC', 'ERC', 'FC', 'FC', 'CC', 'VIS'], 
                '566': ['HPC', 'HPC', 'AMY', 'FC', 'FC', 'PARS', 'CC', 'FUS', 'LTC'], 
                '567': ['HPC', 'HPC', 'AMY', 'PHC', 'ERC'], 
                '568': ['HPC', 'HPC', 'AMY', 'AMY', 'FC', 'FC', 'FC', 'FC', 'CC', 'CC'], 
                '570': ['HPC', 'HPC', 'AMY', 'AMY', 'PHC', 'ERC', 'FC', 'CC', 'INS', 'INS'], 
                '572': ['HPC', 'HPC', 'AMY', 'AMY', 'ERC', 'ERC', 'PARS', 'CC', 'CC'], 
                '573': ['HPC', 'PHC', 'FC', 'FC', 'INS'], 
                'i717': ['AMY', 'ERC', 'ERC', 'FC', 'CC', 'CC', 'MCC', 'FUS', 'INS', 'INS', 'INS', 'INS'], 
                'i728': ['HPC', 'HPC', 'HPC', 'AMY', 'ERC', 'FC', 'FC', 'FC', 'FUS', 'INS', 'LTC', 'VIS'],
                }

SPIKE_CHANNEL = {"555": 37, "562": 56, "563": 64, "564": 56, "565": 80, '566': 72, '567': 40, '568': 80, '570': 80, '572': 72, '573': 40,  '579': 88, '580': 64, '582': 96, 'i717': 96, 'i728': 96}  # "565": 80, '566': 72
SPIKE_FRAME = {"555": 24, "562": 50, "563": 50, "564": 50, "565": 50, '566': 50, '567': 50, '568': 50, '570': 50, '572': 50, '573': 50, '579': 50, '580': 50, '582': 50, 'i717': 50, 'i728': 50}  # 8, 15, 24

LABELS8 = ['WhiteHouse', 'CIA', 'Sacrifice', 'Handcuff', 'Jack', 'Bill', 'Fayed', 'Amar']
