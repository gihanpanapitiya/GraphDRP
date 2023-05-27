import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import datetime
import argparse
from preprocess import save_mix_drug_cell_matrix
from preprocess import save_mix_drug_cell_matrix_candle
import candle
import improve_utils
import urllib
from sklearn.metrics import mean_squared_error

file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'model',
     'type': int,
     'help': '...'
     },
    {'name': 'train_batch',
     'type': int,
     'help': '...'
     },
    {'name': 'val_batch',
     'type': int,
     'help': '...'
     },
    {'name': 'test_batch',
     'type': int,
     'help': '...'
     },
    {'name': 'lr',
     'type': float,
     'help': '....'
     },
    {'name': 'num_epoch',
     'type': int,
     'help': '..'
     },
    {'name': 'log_interval',
     'type': int,
     'help': '.....'
     },
    {'name': 'cuda_name',
     'type': str,
     'help': '...'
     },
    {'name': 'data_url',
     'type': str,
     'help': '...'
     },
    {'name': 'download_data',
     'type': bool,
     'help': '...'
     },
    {'name': 'data_split_seed',
     'type': int,
     'help': '.....'
     },
     {'name': 'data_type',
     'type': str,
     'help': '.....'
     } 
]

required = None


# if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
#     print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
#     cuda_name = os.getenv("CUDA_VISIBLE_DEVICES")
# else:
#     cuda_name = 0

# CUDA_ID = int(cuda_name)
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()




def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, 
           cuda_name, data_path, output_path, dataset_type):

    
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    model_st = modeling.__name__
    dataset = dataset_type
    train_losses = []
    val_losses = []
    val_pearsons = []
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = data_path+'/data/processed/' + dataset + '_train_mix'+'.pt'
    processed_data_file_val = data_path+'/data/processed/' + dataset + '_val_mix'+'.pt'
    processed_data_file_test = data_path+'/data/processed/' + dataset + '_test_mix'+'.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root=data_path+'/data', dataset=dataset+'_train_mix')
        val_data = TestbedDataset(root=data_path+'/data', dataset=dataset+'_val_mix')
        test_data = TestbedDataset(root=data_path+'/data', dataset=dataset+'_test_mix')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False, drop_last=False)
        print("CPU/GPU: ", torch.cuda.is_available())
                
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print(device)
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1
        model_file_name = output_path+'/model_' + model_st + '_' + dataset +  '.model'
        result_file_name = output_path+'/result_' + model_st + '_' + dataset +  '.csv'
        loss_fig_name = output_path+'/model_' + model_st + '_' + dataset + '_loss'
        pearson_fig_name = output_path+'/model_' + model_st + '_' + dataset + '_pearson'
        for epoch in range(num_epoch):
            train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
            G,P = predicting(model, device, val_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
                        
            G_test,P_test = predicting(model, device, val_loader)
            ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret_test)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_pearson = ret[2]
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
            else:
                print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)
        draw_loss(train_losses, val_losses, loss_fig_name)
        draw_pearson(val_pearsons, pearson_fig_name)

        # load the best model
        model.load_state_dict(torch.load(model_file_name))
        true_test, pred_test = predicting(model, device, test_loader)
        # print(true_test, end='')
        # print(true_test, end='')
        print("rmse: ", mean_squared_error(true_test, pred_test)**.5)

        test_smiles = pd.read_csv(data_path+'/test_smiles2.csv')['smiles'].values
        df_res = pd.DataFrame(np.column_stack([true_test,pred_test, test_smiles]), columns=['true', 'pred', 'smiles'])
        df_res.to_csv(output_path+'/test_predictions.csv', index=False)

def get_data(data_url, cache_subdir, download=True):
    
    # cache_subdir = os.path.join(CANDLE_DATA_DIR, 'SWnet', 'Data')
    
    if download:
        print('downloading data')
        os.makedirs(cache_subdir, exist_ok=True)
        os.system(f'svn checkout {data_url} {cache_subdir}')   
        print('downloading done') 

# if __name__ == "__main__":
def run(opt):
    # parser = argparse.ArgumentParser(description='train model')
    # parser.add_argument('--model', type=int, required=False, default=0,     help='0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet')
    # parser.add_argument('--train_batch', type=int, required=False, default=1024,  help='Batch size training set')
    # parser.add_argument('--val_batch', type=int, required=False, default=1024, help='Batch size validation set')
    # parser.add_argument('--test_batch', type=int, required=False, default=1024, help='Batch size test set')
    # parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    # parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    # parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    # parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')

    # args = parser.parse_args()
    # base_path=os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')

    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][opt['model']]
    train_batch = opt['train_batch']
    val_batch = opt['val_batch']
    test_batch = opt['test_batch']
    lr = opt['lr']
    num_epoch = opt['num_epoch']
    log_interval = opt['log_interval']
    cuda_name = opt['cuda_name']
    output_path = opt['output_dir']

    data_path=os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')
    if opt['data_type'] == 'original':
        data_url = opt['data_url']
        download_data = opt['download_data']
        get_data(data_url, data_path, download_data)
        save_mix_drug_cell_matrix(data_path, opt['data_split_seed'])
    elif opt['data_type'] == 'ccle_candle':
        print('running with candle data....')
        dataset_type='CCLE'
        if not os.path.exists(data_path+'/csa_data'):
            download_csa_data(opt)
        save_mix_drug_cell_matrix_candle(data_path=data_path, data_type='CCLE', metric='ic50', data_split_seed=opt['data_split_seed'])



    main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval,
     cuda_name, data_path, output_path, dataset_type)



class GrapgDRP_candle(candle.Benchmark):

        def set_locals(self):
            if required is not None:
                self.required = set(required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions


def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    graphdrp_params = GrapgDRP_candle(
                            filepath=file_path,
                            defmodel="graphdrp_model.txt",
                                            # defmodel="graphdrp_model_candle.txt",
                            framework="pytorch",
                            prog="GraphDRP",
                            desc="CANDLE compliant GraphDRP",
                                )
    gParameters = candle.finalize_parameters(graphdrp_params)
    return gParameters


def download_csa_data(opt):

    csa_data_folder = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data', 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')

    if not os.path.exists(csa_data_folder):
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
        os.mkdir( splits_dir  )
        os.mkdir( x_data_dir  )
        os.mkdir( y_data_dir  )
    

    for file in ['CCLE_all.txt', 'CCLE_split_0_test.txt', 'CCLE_split_0_train.txt', 'CCLE_split_0_val.txt']:
        urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/splits/{file}',
        splits_dir+f'/{file}')

    for file in ['cancer_mutation_count.txt', 'drug_SMILES.txt','drug_ecfp4_512bit.txt' ]:
        urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/x_data/{file}',
        x_data_dir+f'/{file}')

    for file in ['response.txt']:
        urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/y_data/{file}',
        y_data_dir+f'/{file}')


if __name__ == '__main__':

    opt = initialize_parameters()
    print(opt)
    run(opt)


    print("Done.")
