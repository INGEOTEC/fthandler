import subprocess
import io
import os
import sys
import json
import numpy as np
from fthandler.ft import FastTextHandler, search_params, ConfigSpace
from fthandler import norm
from random import choice
from tempfile import mktemp
import pickle
from argparse import ArgumentParser, SUPPRESS
from sklearn.preprocessing import LabelEncoder


def train_main(args):
    """
    searches for best parameters and creates a fastText model

    Parameters
    ----------

    args: ArgParse
    """

    with open(args.training) as f:
        X = [json.loads(line) for line in f.readlines()]

    y = [x[args.klass] for x in X]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    
    if args.params is None:
        pool = Pool(args.nprocs)
        best_list = search_params(np.array(X), np.array(y), pool,
                                bsize=args.bsize, esize=args.esize, n_splits=args.kfolds, tol=args.tol,
                                text_key=args.text)
    else:
        with open(args.params) as f:
            best_list = json.load(f)

    space = ConfigSpace()
    if args.config is not None:
        s = json.loads(args.config)
        for k, v in s.items():
            if isinstance(v, (list, tuple)):
                setattr(space, k, v)
            else:
                setattr(space, k, [v])

    print('saving best list', file=sys.stderr)
    with open(args.model + '.params', 'w') as f:
        print(json.dumps(best_list, indent=2, sort_keys=True), file=f)

    ft = FastTextHandler(model=args.model, **best_list[0][-1]).fit(X, y)
    with open(args.model + '.pickle', "wb") as f:
        pickle.dump((ft, le), f)


def predict_main(args):
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
    with open(name + ".pickle", "rb") as f:
        ft, le = pickle.load(f)
        ft._modelname = name
        ft.text_key = args.text
    
    def read_print_loop(infile, outfile):
        with ft.predict_prob_loop() as predict_prob:
            while True:
                line = infile.readline()
                if len(line) == 0:
                    break

                if args.raw:
                    d = {args.text: line.rstrip()}
                else:
                    d = json.loads(line)

                hy = predict_prob(d)
                d[args.klass + '_prob'] = hy
                d[args.klass] = le.inverse_transform([np.argmax(hy)])[0]
                print(json.dumps(d, sort_keys=True), file=outfile)

    if args.test == '-':
        infile = sys.stdin
    else:
        infile = open(args.test)
    if args.output == '-':
        outfile = sys.stdout
    else:
        outfile = open(args.output, "w")
    
    try:
        read_print_loop(infile, outfile)
    finally:
        infile.close()
        outfile.close()


def normalize_main(args):
    if args.params is None:
        ft = FastTextHandler(model=None)
    else:
        with open(args.params) as f:
            params = json.load(f)

        if isinstance(params, (list, tuple)):
            params = params[0][-1]  # a best list file

        ft = FastTextHandler(model=None, **params)

    def read_print_loop(f, outfile):
        while True:
            line = f.readline()
            if len(line) == 0:
                break
    
            klass = []
            if args.raw:
                if args.parse_labels is not None:  
                    arr = line.split()
                    while arr[0].startswith('__label__'):
                        klass.append(arr.pop(0))

                    d = {args.text: " ".join(arr)}
                else:
                    d = {args.text: line}
            else:
                d = json.loads(line)
                if args.parse_labels:
                    klass = d[args.klass]
                    if isinstance(klass, (int, str)):
                        klass = ["__label__{0}".format(klass)]
                    else:
                        klass = ["__label__{0}".format(k) for k in klass]
            
            data = ft.normalize_one(d)

            if len(klass) > 0:
                print(" ".join(klass) + ' ' + data, file=outfile)
            else:
                print(data, file=outfile)

    if args.dataset == '-':
        infile = sys.stdin
    else:
        infile = open(args.dataset)
    if args.output == '-':
        outfile = sys.stdout
    else:
        outfile = open(args.output, "w")
    
    try:
        read_print_loop(infile, outfile)
    finally:
        infile.close()
        outfile.close()


def sentence_vectors_main(args):
    name = args.model.replace(".pickle", "")
    with open(name + ".pickle", "rb") as f:
        ft, le = pickle.load(f)
        ft._modelname = name
        ft.text_key = args.text

    def read_print_loop(infile, outfile):
        with ft.sentence_vectors_loop() as sentence_vector:
            while True:
                line = infile.readline()
                if len(line) == 0:
                    break

                if args.raw:
                    d = {args.text: line.rstrip()}
                else:
                    d = json.loads(line)

                d[args.vec] = sentence_vector(d)
                print(json.dumps(d, sort_keys=True), file=outfile)

    if args.dataset == '-':
        infile = sys.stdin
    else:
        infile = open(args.dataset)
    if args.output == '-':
        outfile = sys.stdout
    else:
        outfile = open(args.output, "w")

    try:
        read_print_loop(infile, outfile)
    finally:
        infile.close()
        outfile.close()

if __name__ == '__main__':
    from multiprocessing import Pool
    parser = ArgumentParser(description="FastText handler (hyper-parameter optimization and some specific uses)")
    parser.add_argument("--text", default="text", help="key of the text keyword in training file")
    parser.add_argument("--klass", default="klass", help="key of the klass keyword in training file")
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_train = subparsers.add_parser(
        'train', help='optimizes fastText parameter for the given task'
    )
    parser_train.add_argument("--command", default="train", help=SUPPRESS)
    parser_train.add_argument("training", help="training file; each line is a json per line")
    parser_train.add_argument("-m", "--model", type=str, required=True, help="the prefix to save models and parameters")
    parser_train.add_argument("-p", "--params", type=str, default=None, help="specifies the best parameters file instead of searching for it, skips the optimization step")
    parser_train.add_argument("-c", "--config", type=str, default=None,
    help="modifies the default configuration space to use the given starting values for searching; in json format,"
         "e.g., set `--space '{\"dim\": [3, 10]}'` to start searching word embeddings of dimension 3 and 10")
    parser_train.add_argument("-b", "--bsize", type=int, default=16, help="beam size for hyper-parameter optimization")
    parser_train.add_argument("-e", "--esize", type=int, default=4, help="explore this number of best configurations' neighbors")
    parser_train.add_argument("-k", "--kfolds", type=int, default=3, help="k for the k-fold cross-validation")
    parser_train.add_argument("-n", "--nprocs", type=int, help="number of workers to spawn (it uses multiprocessing module)")
    parser_train.add_argument("-t", "--tol", type=float, default=0.001, help="minimum improvement per iteration")
    parser_predict = subparsers.add_parser(
        'predict', help='prediction of input samples'
    )
    parser_predict.add_argument("--command", default="predict", help=SUPPRESS)
    parser_predict.add_argument("model", help="model file; created with train")
    parser_predict.add_argument("test", default='-', help="test file; each line is a json per line")
    # parser_predict.add_argument("-p", "--prob", default=False, action='store_true', help="add probabilities")
    parser_predict.add_argument("-o", "--output", default='-', help="filename to save predictions")
    parser_predict.add_argument("--raw", default=False, action='store_true', help="raw text is used as input instead of json")
    parser_vectors = subparsers.add_parser(
        'sentence-vectors', help='computes and prints sentence vectors for each input'
    )
    parser_vectors.add_argument("--command", default="sentence-vectors", help=SUPPRESS)
    parser_vectors.add_argument("model", help="model file; created with train")
    parser_vectors.add_argument("dataset", help="input file; each line is a json per line")
    parser_vectors.add_argument("--vec", default="vec", help="key to store computed sentence vector")
    parser_vectors.add_argument("-o", "--output", default='-', help="filename to save output")
    parser_vectors.add_argument("--raw", default=False, action='store_true', help="raw text is used as input instead of json")
    parser_norm = subparsers.add_parser(
        'normalize', help='extracts text from a dataset and normalizes it'
    )
    parser_norm.add_argument("--command", default="normalize", help=SUPPRESS)
    parser_norm.add_argument("dataset", help="input file; each line is a json per line")
    parser_norm.add_argument("-p", "--params", default=None, help="params; created with train or by hand; if not given default parameters will be used")
    parser_norm.add_argument("--no-labels", dest="parse_labels", default=True, action='store_false', help="discard parsing labels (only activated with --raw)")
    parser_norm.add_argument("-o", "--output", default='-', help="filename to save output")
    parser_norm.add_argument("--raw", default=False, action='store_true', help="raw text is used as input instead of json")

    args = parser.parse_args()
    if hasattr(args, "command"):
        if args.command == "train":
            train_main(args)
        elif args.command == "predict":
            predict_main(args)
        elif args.command == "sentence-vectors":
            sentence_vectors_main(args)
        elif args.command == "normalize":
            normalize_main(args)
        else:
            parser.print_help()
            sys.exit(-1)
    else:
        parser.print_help()
        sys.exit(-1)
