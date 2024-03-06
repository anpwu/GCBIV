import os
import random
import torch
import numpy as np
import tensorflow as tf

def log(logfile,str,out=True):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    if out:
        print(str)
        
def cat(data_list, axis=1):
    try:
        output=torch.cat(data_list,axis)
    except:
        output=np.concatenate(data_list,axis)

    return output

def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_tf_seed(seed=2021):
    tf.reset_default_graph()
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)