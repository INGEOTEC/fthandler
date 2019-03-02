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
import pickle
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder


def train(args):
    """
    searches for best parameters and creates a fastText model

    Parameters
    ----------

    args: ArgParse
    """
    with open(args.training) as f:
        X = [json.loads(line) for line in f.readlines()]

    # paramsfile = os.environ.get('params', trainfile + ".params")
    # procs = int(os.environ.get('procs', '8'))

    y = [x[args.klass] for x in X]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    pool = Pool(args.nprocs)

    best_list = search_params(np.array(X), np.array(y), pool,
        bsize=args.bsize, esize=args.esize, n_splits=args.kfolds, text_key=args.text)

    print('saving best list', file=sys.stderr)
    with open(args.model + '.params', 'w') as f:
        print(json.dumps(best_list, indent=2, sort_keys=True), file=f)

    ft = FastTextHandler(model=args.model, **params).fit(X, y)
    with open(args.model + '.pickle', "wb") as f:
        pickle.dump((ft, le), f)


def predict(args):
    """
    Predicts and prints to stdout the prediction of a test file

    Parameters
    ----------
    args: ArgParse

    Returns
    -------
    Nothing. This procedure prints in JSON format the predicted labels to the stdout (updating input dictionaries with a "klass" keyword);
    one prediction per line (one valid JSON per line)
    """
    name = args.model.replace(".pickle", "")
    with open(name, "rb") as f:
        ft, le = pickle.load(f)
        ft.model = name
        ft.text_key = args.text

    with open(args.test) as f:
        Xtest = [json.loads(line) for line in f.readlines()]
    
    hy = le.inverse_transform(ft.predict(Xtest))
    f = sys.stdout   
    for x, _hy in zip(Xtest, hy):
        x[args.klass] = _hy
        print(json.dumps(x, sort_keys=True), file=f)


if __name__ == '__main__':
    from multiprocessing import Pool
    parser = ArgumentParser(description="FastText handler (hyper-parameter optimization and some specific uses)")
    parser.add_argument("--text", default="text", help="key of the text keyword in training file")
    parser.add_argument("--klass", default="klass", help="key of the klass keyword in training file")
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_train = subparsers.add_parser(
        'train', help='optimizes fastText parameter for the given task'
    )
    parser_train.add_argument("training", help="training file; each line is a json per line")
    parser_train.add_argument("-m", "--model", required=True, help="the prefix to save models and parameters")
    parser_train.add_argument("-b", "--bsize", default=32, help="beam size for hyper-parameter optimization")
    parser_train.add_argument("-e", "--esize", default=32, help="explore this number of best configurations' neighbors")
    parser_train.add_argument("-k", "--kfolds", default=3, help="k for the k-fold cross-validation")
    parser_train.add_argument("-n", "--nprocs", help="number of workers to spawn (it uses multiprocessing module)")

    parser_predict = subparsers.add_parser(
        'predict', help='prediction of input samples'
    )
    parser_predict.add_argument("model", help="model file; created with train")
    parser_predict.add_argument("test", help="test file; each line is a json per line")

    args = parser.parse_args()
    
    if hasattr(args, "training"):
        train(args)
    elif hasattr(args, "model"):
        predict(args)
    else:
        print("Usage: {0} train|predict train.json test.json".format(sys.argv[0]), file=sys.stderr)
        sys.exit(-1)
