import os
import torch
import argparse
import pandas as pd
from generator import Simulation

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    argparser = argparse.ArgumentParser(description="Script for generating Toy datasets.")

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
    
    args = argparser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    else:
        device = torch.device("cpu")

    ToyData = Simulation(n=args.num, one=args.one, VX=args.multiply, mV=args.mV, mX=args.mX, mU=args.mU, storage_path=args.storage_path)
    ToyData.run(n=args.num, num_reps=args.num_reps)

    # setPath
    dataDir = f'{args.storage_path}/data/{ToyData.which_benchmark}/{ToyData.which_dataset}/'
    resultDir = f'{args.storage_path}/results/{ToyData.which_benchmark}_{ToyData.which_dataset}/'
    reportDir = f'{args.storage_path}/report/'
    os.makedirs(os.path.dirname(dataDir), exist_ok=True)
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    os.makedirs(os.path.dirname(reportDir), exist_ok=True)
