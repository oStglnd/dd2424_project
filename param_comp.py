
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
            # 'units':params['units'],
            'layers':len(params['units']),
            'trainLoss':np.mean(params['trainLossHist'][-50:]) / params['seq_length'],
            'valLoss':np.mean(params['valLossHist'][-50:]) / params['seq_length']
        }
    


# make DF
plotData = pd.DataFrame(dataDict).T

# # get plotVals
layers = plotData.groupby(['layers', 'embeddings']).mean()['trainLoss']
layers = layers.reset_index()


g = sns.catplot(
    data=layers, 
    kind="bar",
    x="layers", 
    y="trainLoss", 
    hue="embeddings",
    palette="Blues", 
    alpha=1.0, 
    height=6
)

plt.ylim(1.4, 2.2)
plt.xlabel('Number of layers')
plt.ylabel('Loss', rotation=0, labelpad=15)
plt.title('Mean loss for last 5000 steps')
plt.savefig(plot_path + 'compEmbedd', dpi=200)
plt.show()

# # get plotVals
layers = plotData.groupby(['layers', 'seq_length']).mean()['trainLoss']
layers = layers.reset_index()


g = sns.catplot(
    data=layers, 
    kind="bar",
    x="layers", 
    y="trainLoss", 
    hue="seq_length",
    palette='rocket', 
    alpha=1.0, 
    height=6
)

plt.ylim(1.4, 2.2)
plt.xlabel('Number of layers')
plt.ylabel('Loss', rotation=0, labelpad=15)
plt.title('Mean loss for last 5000 steps')
plt.savefig(plot_path + 'compSeqLen.png', dpi=200)
plt.show()