
import os
import json

from train_network_w2v_v2 import trainNetwork

# get paths
home_path = os.getcwd()
results_path = home_path + '/results/'

# set filename
fname = 'training_v2'
fpath = results_path + fname


# init dictionary for saving
saveDict = {
    'model_v10':{
        'n_epochs':2,
        'embedding_dim':32,
        'seq_length':25,
        'units':[64]
    },
    'model_v11':{
        'n_epochs':2,
        'embedding_dim':32,
        'seq_length':25,
        'units':[64, 32]
    },
    'model_v12':{
        'n_epochs':2,
        'embedding_dim':32,
        'seq_length':25,
        'units':[64, 64, 32]
    },
    'model_v13':{
        'n_epochs':2,
        'embedding_dim':32,
        'seq_length':25,
        'units':[64, 64, 64, 32]
    },
}

for model, params in saveDict.items():
        # try: 
        #     if saveDict[model]['lossHist']:
        #         continue
        # except KeyError:
        #     pass
    
        n_epochs = params['n_epochs']
        embedding_dim = params['embedding_dim']
        seq_length = params['seq_length']
        units = params['units']
    
        print('\n TRAIN NETWORK ({}): epochs: {}, seq_len: {}, embedding dims: {}, n. layers: {}\n'.format(
            model,
            n_epochs,
            seq_length,
            embedding_dim,
            len(units)
        ))
        
        # get training results
        trainLossHist, valLossHist = trainNetwork(
            n_epochs=n_epochs,
            embedding_dim=embedding_dim,
            seq_length=seq_length,
            units=units,
            model_name=model
        )
        
        # save version-specific results in dictionary
        saveDict[model]['trainLossHist'] = trainLossHist
        saveDict[model]['valLossHist'] = valLossHist
        
        # dump results to JSON
        with open(fpath, 'w') as fp:
            json.dump(saveDict, fp)