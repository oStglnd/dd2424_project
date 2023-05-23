
import os
import json

from train_network_w2v_v2 import trainNetwork

# get paths
home_path = os.getcwd()
results_path = home_path + '\\results\\'

# set filename
fname = 'training_v1'
fpath = results_path + fname


# init dictionary for saving
saveDict = {
    'model_v1':{
        'n_epochs':5,
        'embedding_dim':64,
        'seq_length':50,
        'units':[128]
    },
    'model_v2':{
        'n_epochs':5,
        'embedding_dim':128,
        'seq_length':50,
        'units':[256]
    },
    'model_v3':{
        'n_epochs':5,
        'embedding_dim':256,
        'seq_length':50,
        'units':[512]
    },
    'model_v4':{
        'n_epochs':5,
        'embedding_dim':64,
        'seq_length':100,
        'units':[128]
    },
    'model_v5':{
        'n_epochs':5,
        'embedding_dim':128,
        'seq_length':100,
        'units':[256]
    },
    'model_v6':{
        'n_epochs':5,
        'embedding_dim':256,
        'seq_length':100,
        'units':[512]
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