import subprocess
import io
import os
import sys
import json
import numpy as np
import fthandler.norm as norm
from random import choice
from tempfile import mktemp
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pickle
from contextlib import contextmanager


class ConfigSpace(object):
    """
    Defines the configuration space of fastText
    """
    def __init__(self):
        self.mask_users = [True, False]
        self.mask_urls = [True, False]
        self.mask_nums = [True, False]
        self.mask_hashtags = [True, False]
        self.mask_rep = [True, False]
        self.lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.epoch = [3, 10, 30, 100]
        self.dim = [3, 10, 30, 100]
        self.ws = [2, 3, 4, 5, 6, 7, 8, 9]
        self.wn = [1, 2, 3]
        self.minn = [0, 1, 2, 3]
        self.maxn = [3, 5, 7]

    def param_neighbors(self, field, value):
        """
        Defines perturbations for each parameter
        """
        if field.startswith('mask'):
            return [not value]

        if field in ('wn', 'ws'):
            l = [value + 1, value - 1]
            if l[-1] == 0:
                l.pop()
        elif field in ('minn', 'maxn'):
            l = [value + 1, value - 1]
            if l[-1] == -1:
                l.pop()
        elif field in ('epoch', 'dim'):
            l = [value + 5, value - 5]
            if l[-1] <= 0:
                l.pop()
        elif field == 'lr':
            l = [value + 0.03, value - 0.03]
            if l[-1] <= 0:
                l.pop()
        else:
            raise Exception("Unknown field " + field)
        
        return l

    def random_config(self):
        """
        Returns a random configuration for fastText
        """
        c = {}
        for field, values in self.__dict__.items():
            if field[0] == '_':
                continue

            c[field] = choice(values)        

        return c

    def neighborhood(self, config):
        """
        Computes the neighborhood of the given configuration (based on `param_neighbors`)
        
        Parameters
        ----------
        
        config: dict
           A valid configuration for FastTextHandler

        Returns
        -------
        It returns an iterator of configurations pretty similar to input `config`
          
        """
        for field, values in self.__dict__.items():
            if field[0] == '_':
                continue

            if len(values) == 1:
                continue

            c = config.copy()
            for s in self.param_neighbors(field, config[field]):
                c[field] = s
                yield c


def config_id(c):
    """
    Generates a unique name for each configuration
    
    Parameters
    ----------
    
    c: dict
       A valid configuration for FastTextHandler

    Returns
    -------

    A name for the input configuration `c`
    
    """
    h = sorted(c.items(), key=lambda x: x[0])
    return "_".join(["{0}={1}".format(k, v) for k, v in h if k[0] != '_'])


fastTextPath = "./fastText/fasttext"

if not os.path.exists(fastTextPath):
    raise Exception("Not found a valid fasttext binary in the current directory")


class FastTextHandler(object):
    """
    The fastText handler

    Parameters
    ----------
    mask_users: bool
       Specifies if user mentions are conflated as `_usr` or left untouched
    mask_urls: bool
       Specifies if url occurrences are conflated as `_url` or left untouched
    mask_nums: bool
       Specifies if number occurrences are conflated as `_num` or left untouched
    mask_hashtags: bool
       Specifies if hashtags occurrences are conflated as `_htag` or left untouched
    mask_rep: bool
       Specifies if consecutive repetitions of a single character more than three times are
       conflated to `*` or left untouched
    wn: int
       controls the `wordNgrams` hyper-parameter of fastText (maximum length of word n-grams)
    lr: float
        controls the `lr` hyper-parameter of fastText (learning rate)
    epoch: int
        controls the `epoch` hyper-parameter of fastText (number of epochs in the learning process)
    dim: int
       controls the `dim` hyper-parameter of fastText (the dimension of word-embedding vectors)
    ws: int
       controls the `ws` hyper-parameter of fastText (the window size for learning embeddings)
    minn: int
       controls the `minn` hyper-parameter of fastText (minimum length of character n-grams for subwords)
    maxn: int
       controls the `maxn` hyper-parameter of fastText (maximum length of character n-grams for subwords)
    tmp: str
       the temporary directory to be used for models
    text_key: str
       the name of the keyword to access textual data in input examples
    """
    def __init__(self,
                 mask_users=True, mask_urls=True, mask_nums=True, mask_hashtags=True, mask_rep=True,
                 wn=2, lr=0.1, epoch=10, dim=100, ws=5, minn=0, maxn=0,
                 tmp=".", text_key="text", model=None):
        
        self.wn = wn
        self.mask_users = mask_users
        self.mask_urls = mask_urls
        self.mask_nums = mask_nums
        self.mask_hashtags = mask_hashtags
        self.mask_rep = mask_rep
        self.lr = lr
        self.epoch = epoch
        self.dim = dim
        self.ws = ws
        self.minn = minn
        self.maxn = maxn
        self.tmp = tmp
        self.text_key = text_key
        self._modelname = model

    def get_name(self, prefix):
        """
        Generates a filename for the current configuration


        Parameters
        ----------
        prefix: str
           commonly, the output directory of the output model

        Returns
        -------
        A file's path for the fastText models
        """
        return prefix + config_id(self.__dict__)

    def normalize_one(self, x, label=None):
        """
        Normalizes a single text

        Parameters
        ----------
        x: dict
           A dictionary containing textual data (keyword specified in `self.text_key`)
        label: int or str or None
           An integer or string label associated to `text`

        Returns
        -------

        Returns the normalized text prepared to be used as fastText's input
        """
        text = x[self.text_key]
        text = norm.normalize(
            text,
            mask_users=self.mask_users,
            mask_urls=self.mask_urls,
            mask_nums=self.mask_nums,
            mask_hashtags=self.mask_hashtags,
            mask_rep=self.mask_rep
        )
        
        if label is None:
            return text
        
        s = []
        if isinstance(label, (tuple, list)):
            for l in label:
                s.append("__label__{}".format(l))
        else:
            s.append("__label__{}".format(label))
        
        s.append(text)
        return " ".join(s)
    
    def normalize_array(self, X, y=None):
        """
        Creates a fastText's input from the given classification task

        Parameters
        ----------
        X: an array of dict objects
           A list of items to send as fastText's input
        y: an array of labels, optional
           A list of labels associated to `X`

        Returns
        -------
        A valid fastText input data
        """
        f = io.StringIO()
        for i, x in enumerate(X):
            if y is None:
                label = None
            else:
                label = y[i]

            print(self.normalize_one(x, label=label), file=f)
        
        return f.getvalue()
        
    def fit(self, X, y):
        """
        Creates the model for the given training set (`X`, `y`)

        Parameters
        ----------
        X: an array of dict objects
           A list of dictionaries; each one contains the "text" keyword to access the textual data (see `text_key`)
        y: an array of labels
           A list of labels associated to `X`

        Returns
        -------

        The self object
        """
        if self._modelname is None:
            name = self.get_name("model.")
            self._modelname = mktemp(prefix=name, dir=self.tmp)

        data = self.normalize_array(X, y)
        # currently fasttext doesn't support training from stdin
        train = self._modelname + ".input"
        try:
            with open(train, "w") as f:
                f.write(data)

            args = [fastTextPath, "supervised", "-input", train, "-output", self._modelname, "-wordNgrams", str(self.wn),
                    "-lr", str(self.lr), "-epoch", str(self.epoch), "-dim", str(self.dim), "-ws", str(self.ws),
                    "-minn", str(self.minn), "-maxn", str(self.maxn), "-thread", "1"]
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
            outs, errs = proc.communicate()

            if not os.path.isfile(self._modelname + ".bin"):
                raise Exception("An error arised while creating a FT model with the following command:\n{0}".format(" ".join(args)))
        finally:
            os.unlink(train)
        
        print('finished', self._modelname, file=sys.stderr)
        return self

    def delete_model_file(self):
        """
        Deletes the underlying model file (internal method)
        """
        for e in ('.bin', '.vec'):
            try:
                os.unlink(self._modelname + e)
            except Exception as e:
                print("A problem was found while removing model files {0}".format(self._modelname), file=sys.stderr)
                print(e, file=sys.stderr)

    @contextmanager
    def predict_prob_loop(self):
        """
        Creates a context manager ("with" syntax) to evaluate predict probabilities. The function created receives a unique parameter `X`
        which is a list of dictionaries; each one contains the "text" keyword to access the textual data (see `text_key`). The function returns
        the associated probabilities for each class for each item 
        
        Returns
        -------
        A context manager

        """
        proc = subprocess.Popen([fastTextPath, "predict-prob", self._modelname + ".bin", "-", "1000"],
                                stdin=subprocess.PIPE, encoding='utf8', stdout=subprocess.PIPE)

        def predict(x):
            data = self.normalize_one(x)
            proc.stdin.write(data)
            proc.stdin.write('\n')
            proc.stdin.flush()
            line = proc.stdout.readline()
            L = norm.encode_predict_prob(line)
            # print((x, data, line, L), file=sys.stderr)
            return L

        yield predict
        proc.kill()

    def predict_prob(self, X):
        """
        Computes class's probabilities for each item in `X` using predict-prob

        Parameters
        ----------
        X: an array of dict objects
           A list of dictionaries; each one contains the "text" keyword to access the textual data (see `text_key`)
 
        Returns
        -------
        
        Returns a list of vectors in input's order
        """
        with self.predict_prob_loop() as predict_prob:
            return [predict_prob(w) for w in X]
        
    def predict(self, X):
        """
        Computes the label's prediction for each item in `X`

        Parameters
        ----------
        X: an array of dict objects
           A list of dictionaries; each one contains the "text" keyword to access the textual data (see `text_key`)
 
        Returns
        -------
        Returns a list of labels in input's order

        """
        probs = self.predict_prob(X)
        # print("XXXX LEN:", len(probs), np.unique([len(p) for p in probs]), file=sys.stderr)
        return np.argmax(probs, axis=1)

    @contextmanager
    def sentence_vectors_loop(self):
        """
        Creates a function to compute sentence vectors (use "with" syntax).
        
        """
        cmd = [fastTextPath, "print-sentence-vectors", self._modelname + ".bin"]
        print(cmd, file=sys.stderr)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, encoding='utf8', stdout=subprocess.PIPE)

        def fun(x):
            data = self.normalize_one(x)
            proc.stdin.write(data)
            proc.stdin.write('\n')
            proc.stdin.flush()
            return [float(d) for d in proc.stdout.readline().split()]

        yield fun
        proc.kill()

    def sentence_vectors(self, X):
        """
        Returns sentence vectors for each item in X
        """
        with self.sentence_vectors_loop() as vector:
            return [vector(w) for w in X]

    
def run_one(config, X, y, Xtest, ytest, text_key):
    """
    Measures the performance of fastText with a given configuration, training set and validation set

    Parameters
    ----------
    config: dict
       A valid configuration for FastTextHandler
    X: an array of dict objects
       Training set. A list of dictionaries; each one contains the "text" keyword to access the textual data (see `text_key`)
    y: an array of labels
       A list of labels associated to `X`
    Xtest: an array of dict objects
       Validation set. Idem `X`
    ytest: an array of labels
       A list of labels associated to `Xtest`
    text_key: str
       The keyword to accessing textual data in `X`
    Returns
    -------
    The score (currently macro-recall)
    """
    try:
        ft = FastTextHandler(text_key=text_key, **config).fit(X, y)
        hy = ft.predict(Xtest)
        score = recall_score(ytest, hy, average='macro')
        # score = (np.array(ytest) == np.array(hy)).mean()
        # score = f1_score(le.transform(ydev), le.transform(hy), average='binary')
    finally:
        ft.delete_model_file()
        
    return score


def run_kfold(data):
    """
    Measures the performance of fastText on the tuple `data` using k-folds
    This function was designed to be used in multiprocessing mapping
    
    Parameters
    ----------
    - data: tuple
        Contains the configuration for FastTextHandler, the training set (X, y) and the kfolds partitioning
        
    config: dict
       A valid configuration for FastTextHandler
    X: an array of dict objects
       Training set. A list of dictionaries; each one contains the "text" keyword to access the textual data (see `text_key`)
    y: an array of labels
       A list of labels associated to `X`,
    text_key: str
       The key for accessing textual data in `X`

    Returns
    -------
    The score, i.e., average of all folds

    """
    config, Xfull, yfull, kfolds, text_key = data
    # kfolds = StratifiedKFold(n_splits=n_splits)

    score = 0.0
    for train_index, val_index in kfolds.split(Xfull, yfull):
        score += run_one(config, Xfull[train_index], yfull[train_index], Xfull[val_index], yfull[val_index], text_key)

    return score / kfolds.get_n_splits(), config


def search_params(X, y, pool, bsize=32, esize=4, n_splits=3, tol=0.001, text_key='text', space=ConfigSpace()):
    """
    Searches for the best fastText configuration on `X` and `y` using beam-search meta-heuristic

    Parameters
    ----------
    
    X: an array of dictionaries
       examples for training (dictionaries containing text)
    y: an array of labels
       labels associated to `X`
    pool: multiprocessing.Pool
       a pool of workers
    bsize: int
       a hyper-parameter that controls the beam's size (how many best configurations are stored along the search process)
    esize: int
       a hyper-paramter  that controls how many best items (among the beam) are explored at each iteration
    n_splits: int
       determines the number of folds to measure the score
    tol: float
       minimum improvement tolerance for the optimization process
    text_key: str
       the key used to access textual data in each item of `X`
    space: ConfigSpace
       specifies the search space of configurations
    """
    if space is None:
        space = ConfigSpace()
    
    tabu = set()
    data_list = []
    kfolds = StratifiedKFold(n_splits, shuffle=True)

    for i in range(bsize):
        c = space.random_config()
        h = config_id(c)
        if h in tabu:
            continue

        tabu.add(h)
        # data_list.append((c, X, y, Xtest, ytest))
        data_list.append((c, X, y, kfolds, text_key))

    prev = 0
    curr = 0
    best_list = []
    while len(data_list) > 0:
        print("***************\n*** Starting an iteration with {0} candidates".format(len(data_list)), file=sys.stderr)
        e = pool.map(run_kfold, data_list)
        best_list.extend(e)
        best_list.sort(key=lambda x: x[0], reverse=True)

        curr = best_list[0][0]
        if abs(curr - prev) <= tol:
            break

        with open('best_list.json.tmp', 'w') as f:
            print(json.dumps(best_list, indent=2, sort_keys=True), file=f)
            print("*** Current best configuration:", json.dumps(best_list[0], sort_keys=True), file=sys.stderr)

        prev = curr
        data_list = []
        for i in range(min(esize, len(best_list))):
            for c in space.neighborhood(best_list[i][-1]):
                h = config_id(c)
                if h in tabu:
                    continue

                tabu.add(h)
                # data_list.append((c, X, y, Xtest, ytest))
                data_list.append((c, X, y, kfolds, text_key))

        if len(data_list) > bsize:
            np.random.shuffle(data_list)
            data_list = data_list[:bsize]
        
        data_list

    return best_list
