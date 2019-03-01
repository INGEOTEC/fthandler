import subprocess
import io
import os
import sys
import json
import numpy as np
import fthandler.norm
from random import choice
from tempfile import mktemp
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class ConfigSpace(object):
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
        c = {}
        for field, values in self.__dict__.items():
            if field[0] == '_':
                continue

            c[field] = choice(values)        

        return c

    def neighborhood(self, config):
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
    h = sorted(c.items(), key=lambda x: x[0])
    return "_".join(["{0}={1}".format(k, v) for k, v in h if k[0] != '_'])


fastTextPath = "./fasttext"

if not os.path.exists(fastTextPath):
    raise Exception("Not found a valid fasttext binary in the current directory")


class FastTextHandler(object):
    def __init__(self,
                 mask_users=True, mask_urls=True, mask_nums=True, mask_hashtags=True, mask_rep=True,
                 wn=2, lr=0.1, epoch=10, dim=100, ws=5, minn=0, maxn=0,
                 tmp=".", text_key="text"):
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

    def get_name(self, prefix):
        return prefix + config_id(self.__dict__)

    def normalize_one(self, text):
        _text = norm.normalize(
            text,
            mask_users=self.mask_users,
            mask_urls=self.mask_urls,
            mask_nums=self.mask_nums,
            mask_hashtags=self.mask_hashtags,
            mask_rep=self.mask_rep
        )
        return _text
    
    def normalize_as_input(self, X, y=None):
        f = io.StringIO()
        for i, x in enumerate(X):
            if y is None:
                label = None
            else:
                label = y[i]

            text = x[self.text_key]
            text = self.normalize_one(text)

            if label is None:
                print(text, file=f)
            else:
                print("__label__{0} {1}".format(label, text), file=f)
        
        return f.getvalue()

    def load(self, model):
        self._model = model
        
    def fit(self, X, y):
        name = self.get_name("model.")
        self._model = mktemp(prefix=name, dir=self.tmp)
        data = self.normalize_as_input(X, y)
        # currently fasttext doesn't support training from stdin
        train = self._model + ".input"
        with open(train, "w") as f:
            f.write(data)

        args = [fastTextPath, "supervised", "-input", train, "-output", self._model, "-wordNgrams", str(self.wn), "-lr", str(self.lr), "-epoch", str(self.epoch),
                "-dim", str(self.dim), "-ws", str(self.ws), "-minn", str(self.minn), "-maxn", str(self.maxn), "-thread", "1"]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
        outs, errs = proc.communicate()

        if not os.path.isfile(self._model + ".bin"):
            raise Exception("An error arised while creating a FT model with the following command:\n{0}".format(" ".join(args)))

        os.unlink(train)
        print('finished', name, file=sys.stderr)
        return self

    def delete_model_file(self):
        for e in ('.bin', '.vec'):
            try:
                os.unlink(self._model + e)
            except Exception as e:
                print("A problem was found while removing model files {0}".format(self._model), file=sys.stderr)
                print(e, file=sys.stderr)
                
    def decision_function(self, X):
        test = self.normalize_as_input(X)
        dfun = subprocess.check_output([fastTextPath, "predict-prob", self._model + ".bin", "-", "10000"], input=test, encoding='utf8')
        dfun = norm.encode_prediction(dfun.split('\n'))
        return dfun

    def predict(self, X):
        data = self.normalize_as_input(X)
        proc = subprocess.Popen([fastTextPath, "predict", self._model + ".bin", "-", "1"],
                                stdin=subprocess.PIPE, encoding='utf8', stdout=subprocess.PIPE)
        outs, errs = proc.communicate(input=data)
        out = [int(line.replace('__label__', '')) for line in outs.split('\n') if len(line) > 0]
        return out


def run_one(config, X, y, Xtest, ytest):
    ft = FastTextHandler(**config).fit(X, y)
    hy = ft.predict(Xtest)
    score = recall_score(ytest, hy, average='macro')
    # score = (np.array(ytest) == np.array(hy)).mean()
    # score = f1_score(le.transform(ydev), le.transform(hy), average='binary')
    ft.delete_model_file()
    return score


def run_kfold(data):
    config, Xfull, yfull, kfolds = data
    # kfolds = StratifiedKFold(n_splits=n_splits)

    score = 0.0
    for train_index, val_index in kfolds.split(X, y):
        score += run_one(config, Xfull[train_index], yfull[train_index], Xfull[val_index], yfull[val_index])

    return score / kfolds.get_n_splits(), config


def search_params(X, y, pool, bsize=32, esize=4, n_splits=3):
    space = ConfigSpace()
    tabu = set()
    data_list = []
    kfolds = StratifiedKFold(n_splits)

    for i in range(bsize):
        c = space.random_config()
        h = config_id(c)
        if h in tabu:
            continue

        tabu.add(h)
        # data_list.append((c, X, y, Xtest, ytest))
        data_list.append((c, X, y, kfolds))

    prev = 0
    curr = 0
    best_list = []
    while len(data_list) > 0:
        print("***************\n*** Starting an iteration with {0} candidates".format(len(data_list)), file=sys.stderr)
        e = pool.map(run_kfold, data_list)
        best_list.extend(e)
        best_list.sort(key=lambda x: x[0], reverse=True)

        curr = best_list[0][0]
        if curr == prev:
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
                data_list.append((c, X, y, kfolds))

        if len(data_list) > bsize:
            np.random.shuffle(data_list)
            data_list = data_list[:bsize]
        
        data_list

    return best_list
