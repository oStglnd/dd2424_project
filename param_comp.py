
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# get paths
home_path = os.getcwd()
results_path = home_path + '/results/'
plot_path = home_path + '/plots/'

# set filename
fnames = [
    'training_v1',
    'training_v2',
    # 'training_v3'
]

dataDict = {}
for fname in fnames:
    fpath = results_path + fname
    
    with open(fpath, 'r') as fp: 
        results = json.load(fp)
    
    for model, params in results.items():
        
        dataDict[model] = {
            'embeddings':params['embeddings'],
            'embedding_dim':params['embedding_dim'],
            'seq_length':params['seq_length'],
            'units':params['units'],
            'trainLoss':np.mean(params['trainLossHist'][-1000:]),
            'valLoss':np.mean(params['valLossHist'][-1000:])
        }
    


# make DF
plotData = pd.DataFrame(dataDict).T

# # get plotVals
# pDropAcc = plotData.groupby('pDrop').mean()['acc'] * 100
# mAcc = plotData.groupby('m').mean()['acc'] * 100

# # plot dropout accuracy
# sns.barplot(
#     x=pDropAcc.index, 
#     y=pDropAcc.values, 
#     palette='magma'
# )

# plt.ylim(45, 55)
# plt.xlabel('Dropout rate, $p_{drop}$')
# plt.ylabel('%')
# plt.title('Mean accuracy per dropout rates')
# plt.savefig(plot_path + 'compDropout.png', dpi=400)
# plt.show()

# # plot m accuracy
# sns.barplot(
#     x=mAcc.index.astype('int64'), 
#     y=mAcc.values, 
#     palette='viridis'
# )

# plt.ylim(45, 55)
# plt.xlabel('Number of hidden nodes, $m$')
# plt.ylabel('%')
# plt.title('Mean accuracy per layer width')
# plt.savefig(plot_path + 'compLayerM.png', dpi=400)
# plt.show()