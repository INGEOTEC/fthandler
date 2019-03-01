import subprocess
import io
import os
import sys
import json
import numpy as np
from fthandler.ft import FastTextHandler, search_params
from fthandler import norm
from random import choice
from tempfile import mktemp
from sklearn.preprocessing import LabelEncoder


klass_key = os.environ.get('klass', 'klass')
text_key = os.environ.get('text', 'text')


def predict_test(X, y, Xtest, params, le):
    """
    Predicts and prints to stdout the prediction of ::params::Xtest using a fastText model created with `X`, `y`,
    and `params` (fastText's hyper-parameters). 

    Parameters
    ----------
    - X: an array of dictionaries, each one with a "text" keyword to access the actual textual data
       Examples for training
    - y: an array of labels (integers or strings)
       Labels associated to `X`
    - Xtest: idem to `X`
       Dataset to be predicted
    - params: dict
       Hyper-parameters for the fastText model
    - le: LabelEncoder
       A label encoder to map predicted labels to the original naming

    Returns
    -------
    Nothing. This procedure prints in JSON format the predicted labels to the stdout (updating input dictionaries with a "klass" keyword);
    one prediction per line (one valid JSON per line)
    
    """
    ft = FastTextHandler(**params).fit(X, y)
    hy = le.inverse_transform(ft.predict(Xtest))

    f = sys.stdout
    for x, _hy in zip(Xtest, hy):
        xk[lass_key] = _hy
        print(json.dumps(x, sort_keys=True), file=f)
    

if __name__ == '__main__':
    from multiprocessing import Pool
    
    if len(sys.argv) < 3:
        cmd = ''
    else:
        cmd = sys.argv[1]

    if cmd not in ('predict', 'params'):
        print("Usage: python -m fthandler predict train.json test.json", file=sys.stderr)
        print("Usage: python -m fthandler params train.json", file=sys.stderr)
        sys.exit(-1)
        
    trainfile = sys.argv[2]
 
    with open(trainfile) as f:
        X = [json.loads(line) for line in f.readlines()]

    paramsfile = os.environ.get('params', trainfile + ".params")
    procs = int(os.environ.get('procs', '8'))

    y = [x[klass_key] for x in X]
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    if cmd == "predict":
        testfile = sys.argv[3]

        with open(testfile) as f:
            Xtest = [json.loads(line) for line in f.readlines()]

        with open(paramsfile) as f:
            config = json.load(f)[0][-1]
            predict_test(X, y, Xtest, config, le)
    
    elif cmd == "params":
        pool = Pool(procs)
        bsize = int(os.environ.get("bsize", "32"))
        esize = int(os.environ.get("esize", "4"))
        kfolds = int(os.environ.get("kfolds", "3"))
 
        best_list = search_params(np.array(X), np.array(y), pool, bsize=bsize, esize=esize, n_splits=kfolds)
        
        print('saving best list into', paramsfile, file=sys.stderr)
        with open(paramsfile, 'w') as f:
            print(json.dumps(best_list, indent=2, sort_keys=True), file=f)
    
    else:
        print("Usage: {0} predict|params train.json test.json".format(sys.argv[0]), file=sys.stderr)
        sys.exit(-1)
