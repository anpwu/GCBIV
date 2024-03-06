import os
import argparse
import pandas as pd
import numpy as np
import torch

from latent import CausalDataset as VAECausalDataset
from latent import VAEXT, PreProcess
from module import CausalDataset, runGCBIV

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    argparser = argparse.ArgumentParser(description="Script for running GCBIV on Toy datasets.")

    # Run settings
    argparser.add_argument('--seed', default=2021, type=int, help='Random seed for reproducibility.')
    argparser.add_argument('--mode', default='Latent', type=str, choices=['IV', 'OIVL', 'IIVL', 'LIVL', 'Latent'], 
                            help='Mode for the experiment: IV\OIVL\IIVL\LIVL\Latent.')
    argparser.add_argument('--rewrite_log', default=False, type=bool, help='Whether to overwrite existing log files.')
    argparser.add_argument('--use_gpu', default=True, type=bool, help='Flag indicating whether to use GPU if available.')

    # Data settings
    argparser.add_argument('--num', default=10000, type=int, help='Number of samples in the dataset for training, validation, and testing.')
    argparser.add_argument('--num_reps', default=10, type=int, help='Number of repetitions for the dataset.')
    argparser.add_argument('--one', default=1, type=int, help='Flag indicating whether to use full-one weight to generate the dataset.')
    argparser.add_argument('--multiply', default=1, type=int, help='Dimensionality of V * X.')
    argparser.add_argument('--mV', default=2, type=int, help='Dimensionality of instrumental variables V.')
    argparser.add_argument('--mX', default=4, type=int, help='Dimensionality of confounding variables X.')
    argparser.add_argument('--mU', default=4, type=int, help='Dimensionality of unobserved confounding variables U.')
    argparser.add_argument('--storage_path', default='./Data/', type=str, help='Directory to store the data.')
    
    # Latent settings
    argparser.add_argument('--onlyX', default=1, type=int, choices=[0, 1], help='Flag indicating whether only X will be used.')

    # Regression settings
    argparser.add_argument('--batch_size', default=500, type=int, help='Batch size for training.')
    argparser.add_argument('--lr', default=0.05, type=float, help='Learning rate for training.')
    argparser.add_argument('--num_epoch', default=3, type=int, help='Number of epochs for training.')

    args = argparser.parse_args()
    return args


def evaluation(val_dict, synATE):
    train_ATE  = val_dict['ate_train'] - synATE
    test_ATE   = val_dict['ate_test'] - synATE

    return train_ATE, test_ATE


if __name__ == "__main__":

    args = get_args()

    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    else:
        device = torch.device("cpu")

    # Set Paths
    which_benchmark = 'Simulation'
    which_dataset = f'{args.mV}_{args.mX}_{args.mU}'
    dataDir = f'{args.storage_path}/data/{which_benchmark}/{which_dataset}/'
    resultDir = f'{args.storage_path}/results/{which_benchmark}_{which_dataset}/'
    reportDir = f'{args.storage_path}/report/'
    logfile = f'{resultDir}/log.txt'

    if args.rewrite_log:
        f = open(logfile,'w')
        f.close()

    # GCBIV - Recover Latent Variables
    for exp in range(args.num_reps):
        dataPath  = f'{dataDir}/{exp}/'
        savePath = f'{dataDir}/{exp}/Latent/'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)

        data = VAECausalDataset(path=dataPath, num=args.num)
        data.train.pandas(savePath+'train.csv')
        data.valid.pandas(savePath+'val.csv')
        data.test.pandas(savePath+'test.csv')

        model=VAEXT()

        if args.onlyX:
            model.config['x_mode'] = [2,2,2,2]
        else:
            data.train.x = np.concatenate([data.train.v, data.train.x], axis=1)
            data.valid.x = np.concatenate([data.valid.v, data.valid.x], axis=1)
            data.test.x  = np.concatenate([data.test.v,  data.test.x], axis=1)
            model.config['x_mode'] = [2,2,2,2,2,2]
        
        model.config['t_mode'] = [1]
        model.config['y_mode'] = [2]
        model.config['save_path'] = savePath
        model.config['latent_dim'] = 5
        model.config['epochs'] = 5
        model.config['save_per_batch'] = 10
        model.config['show_per_batch'] = 10

        model.fit(data)
        
        trainITE_0,trainITE_1,trainITE_t = model.ITE(data.train)
        validITE_0,validITE_1,validITE_t = model.ITE(data.valid)
        testITE_0, testITE_1, testITE_t  = model.ITE(data.test)

    # GCBIV - Regress Treatment Variables
    for exp in range(args.num_reps):
        data_rst = np.load(dataDir+f'/{exp}/Latent/2_0_result.npz')
        train_df = pd.read_csv(dataDir + f'{exp}/train.csv')
        val_df = pd.read_csv(dataDir + f'{exp}/val.csv')
        test_df = pd.read_csv(dataDir + f'{exp}/test.csv')

        train = CausalDataset(train_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])
        val = CausalDataset(val_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])
        test = CausalDataset(test_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])

        train.v = data_rst['trt_rlt'][:args.num]
        val.v   = data_rst['val_rlt'][:args.num]
        test.v  = data_rst['tst_rlt'][:args.num]
        args.mV = train.v.shape[1]

        train, val, test = PreProcess(exp, args, dataDir, resultDir, train, val, test, device)

    # GCBIV - Regress Outcome Variables
    synATE = 1.0
    GCBIVres = []
    for exp in range(args.num_reps):
        train_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/train.csv')
        val_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/val.csv')
        test_df = pd.read_csv(dataDir + f'{exp}/{args.mode}/test.csv')

        train = CausalDataset(train_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])
        val = CausalDataset(val_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])
        test = CausalDataset(test_df, variables = ['u','x','v','z','p','s','m','t','g','y','f','c'])

        if args.mode == 'Latent':
            data_rst = np.load(dataDir+f'/{exp}/Latent/2_0_result.npz')

            train.v = data_rst['trt_rlt'][:args.num]
            val.v   = data_rst['val_rlt'][:args.num]
            test.v  = data_rst['tst_rlt'][:args.num]
            args.mV = train.v.shape[1]

            train.x = np.concatenate([data_rst['trt_rlt'][:args.num], train.x],1)
            val.x = np.concatenate([data_rst['val_rlt'][:args.num], val.x],1)
            test.x  = np.concatenate([data_rst['tst_rlt'][:args.num], test.x],1)
            args.mX = train.x.shape[1]
            
        mse_val, obj_val, final = runGCBIV(exp, args, dataDir, resultDir, train, val, test, device)

        print(evaluation(mse_val, synATE))
        print(evaluation(obj_val, synATE))
        print(evaluation(final, synATE))

        train_ATE, test_ATE = evaluation(obj_val,synATE)
        GCBIVres.append([train_ATE, test_ATE])
        tmp_dfCBIV = pd.DataFrame(np.array(GCBIVres), columns=['ATE_train', 'ATE_test'])
        tmp_dfCBIV.to_csv(f'./Data/report/Simulation_{args.mode}_GCBIV.csv')
