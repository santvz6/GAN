import numpy as np
import pickle
import inspect

# Monkey patch numpy
np.bool = np.bool_
np.int = np.int64
np.float = np.float64
np.complex = np.complex128
np.object = object
np.unicode = str
np.str = str

# Monkey patch inspect for chumpy
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import chumpy as ch

def convert_model(pkl_path, out_pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    clean_data = {}
    for k, v in data.items():
        if isinstance(v, ch.Ch):
            clean_data[k] = np.array(v)
        else:
            clean_data[k] = v
            
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(clean_data, f)
    print(f"Converted {pkl_path} to clean PKL at {out_pkl_path}")

import sys
if __name__ == '__main__':
    convert_model(sys.argv[1], sys.argv[2])
