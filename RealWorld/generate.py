import os
import argparse
import pandas as pd
import numpy as np
import torch

from utils import IHDP_Generator, Twins_Generator

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    argparser = argparse.ArgumentParser(description="Script for generating Toy datasets.")

    # Run settings
    argparser.add_argument('--seed', default=2024, type=int, help='Random seed for reproducibility.')
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
    argparser.add_argument('--mU', default=2, type=int, help='Dimensionality of unobserved confounding variables U.')
    argparser.add_argument('--storage_path', default='./Data/', type=str, help='Directory to store the data.')
    
    args = argparser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else "cpu")
    else:
        device = torch.device("cpu")

    dataDir='./Data/Causal/'
    storage_path='./Data/'

    IHDP = IHDP_Generator(mV=2, mX=4, mU=2, details=1, dataDir=dataDir, storage_path=storage_path)
    IHDP.run(args.num_reps)

    Twins_553 = Twins_Generator(sc=1, sh=-2, mV=5, mX=5, mU=3, details=1,dataDir=dataDir, storage_path=storage_path)
    Twins_553.run(args.num_reps)

    # setPath
    reportDir = f'{args.storage_path}/report/'
    os.makedirs(os.path.dirname(reportDir), exist_ok=True)