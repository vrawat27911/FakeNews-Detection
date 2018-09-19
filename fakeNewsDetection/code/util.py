from __future__ import division
import os
import _pickle as pickle
import nltk
import sys
import time
import pandas as pd
import numpy as np
from fnc_baseline.utils.score import report_score, LABELS, score_submission
from fnc_baseline.utils.dataset import DataSet
from importlib import reload
reload(sys)


# hyperparameters
class Config:
  n_epochs = 40
  lr = 0.001
  batch_size = 128
  n_classes = 4
  hidden_size = 100
  n_layers = 0
  model = None
  dropout = 0.8
  trainable_embeddings = 'Variable'
  max_length = None
  attention_length = 15
  hidden_next = 0.6

  ## Params assigned later using Glove embeddings:
  embed_size = None
  vocab_size = None
  pretrained_embeddings = []
  num_samples = None
  h_max_len = None  # for headlines
  b_max_len = None  # for body


def split_data(data, prop_train=0.6, prop_dev=0.2, seed=None):
    np.random.seed(seed)
    assert prop_train + prop_dev <= 1

    if (type(data).__module__ == np.__name__):

        num_samples = data.shape[0]
        num_train_samples = int(np.floor(num_samples * prop_train))
        num_dev_samples = int(np.floor(num_samples * prop_dev))

        indices = list(range(num_samples))
        np.random.shuffle(indices)

        train_indices = indices[0:num_train_samples]
        dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
        test_indices = indices[num_train_samples + num_dev_samples:num_samples]

        train_data = data[indices[train_indices], :]
        dev_data = data[indices[dev_indices], :]
        test_data = data[indices[test_indices], :]

    elif isinstance(data, list):

        num_samples = len(data)
        num_train_samples = int(np.floor(num_samples * prop_train))
        num_dev_samples = int(np.floor(num_samples * prop_dev))

        indices = list(range(num_samples))
        np.random.shuffle(indices)

        train_indices = indices[0:num_train_samples]
        dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
        test_indices = indices[num_train_samples + num_dev_samples:num_samples]

        train_data = [data[i] for i in train_indices]
        dev_data = [data[i] for i in dev_indices]
        test_data = [data[i] for i in test_indices]

    return train_data, dev_data, test_data, train_indices, dev_indices, test_indices,


def split_indices(num_samples, prop_train=0.6, prop_dev=0.2):
    num_train_samples = int(np.floor(num_samples * prop_train))
    num_dev_samples = int(np.floor(num_samples * prop_dev))
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    train_indices = indices[0:num_train_samples]
    dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
    test_indices = indices[num_train_samples + num_dev_samples:num_samples]
    return train_indices, dev_indices, test_indices

def pack_labels(data, labels, seqlen):
    output = []
    num_rows = data.shape[0]
    assert num_rows == len(labels)
    for i in range(data.shape[0]):
        the_row = data[i, :]
        output.append((the_row, labels[i], seqlen[i]))
    return output


def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x = x - np.amax(x, axis=1).reshape(x.shape[0], 1)
        rowSums = np.sum(np.exp(x), axis=1).reshape(x.shape[0], 1)
        x = np.exp(x) / rowSums
    else:
        # Vector
        x = x - np.max(x)
        theSum = np.sum(np.exp(x))
        x = np.exp(x) / theSum

    assert x.shape == orig_shape
    return x


# Compute performance metrics
def get_performance(predicted, truth, n_classes=None, outputStyle='dict'):

    predicted = np.asarray(predicted, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)

    assert len(predicted) == len(truth)

    # Compute competition score:
    competition_score = scorer(predicted, truth)
    output = []

    if n_classes is None:
        n_classes = len(np.unique(predicted.extend(truth)))

    for i in range(n_classes):
        tp = sum((predicted == i) & (truth == i))
        tn = sum((predicted != i) & (truth != i))
        fp = sum((predicted == i) & (truth != i))
        fn = sum((predicted != i) & (truth == i))

        print('tp ' + str(tp))
        print('tn ' + str(tn))
        print('fp ' + str(fp))
        print('fn ' + str(fn))

        # Compute performance metrics
        recall = tp / (tp + fn)  # aka sensitivity
        print('recall ' + str(recall))
        precision = tp / (tp + fp)  # aka ppv
        print('precision ' + str(precision))
        specificity = tn / (tn + fp)
        print('specificity ' + str(specificity))
        f1 = 2 * tp / (2 * tp + fp + fn)
        print('f1 ' + str(f1))
        accuracy = (tp + tn) / len(truth)

        keys = ['tp', 'tn', 'fp', 'fn', 'recall', 'precision', 'specificity', 'f1', 'accuracy', 'competition']
        values = [tp, tn, fp, fn, recall, precision, specificity, f1, accuracy, competition_score]
        output.append(dict(zip(keys, values)))

    return output


# Computes competition score
def scorer(pred, truth):
    # Maximum possible score
    max_score = 0.25 * sum(truth == 3) + 1 * sum(truth != 3)
    # Computing achieved sore
    # Score from unrelated correct
    unrelated_score = 0.25 * sum((truth == 3) & (pred == truth))
    # Score from related correct, but specific class incorrect
    related_score1 = 0.25 * sum((truth != 3) & (pred != truth) & (pred != 3))
    # Score from getting related correct, specific class correct
    related_score2 = 0.75 * sum((truth != 3) & (pred == truth))

    final_score = (unrelated_score + related_score1 + related_score2) / max_score
    return final_score

def get_minibatches(data, minibatch_size):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]

        if list_data:
            examples_minibatch = minibatch(data[0], minibatch_indices)
            labels_minibatch = minibatch(data[1], minibatch_indices)
            seqlen_minibatch = minibatch(data[2], minibatch_indices)

            # Truncating sentences to the max_length of the minibatch, placeholders have fixed side
            # max_len_minibatch = max(seqlen_minibatch)
            # examples_minibatch = examples_minibatch[:,:max_len_minibatch]

            yield [examples_minibatch, labels_minibatch, seqlen_minibatch]

        else:  # no truncating if data not in the 'packed' list format [examples, labels, seqlen]
            yield minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size)


# from fchollet keras textbook
class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step). The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step). The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)



# load the data
def read_data(base_path = '/Users/spfohl/Documents/CS_224n/project/altfactcheckers'):
    
    # Extracting data
    dataset = DataSet(path = base_path + '/data')
    stances = dataset.stances
    articles = dataset.articles

    h, b, y = [],[],[]
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    y = np.asarray(y, dtype = np.int64)
    return h, b, y

# fetch glove embeddings
def loadGloVe(filename):

    # embedding dimensions for words from Glove
    file_0 = open(filename,'r')
    line = file_0.readline()
    emb_dim = len(line.strip().split(' ')) - 1
    file_0.close()

    # First row of embedding matrix is 0 for zero padding
    vocab = ['<pad>']
    # embd = [[0.0] * emb_dim]
    embd = np.zeros(shape=[1193515 + 1, emb_dim], dtype=np.float64)

    file = open(filename,'r', errors='ignore')
    i = 1

    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        temp = np.array(row[1:], dtype=np.float64)
        temp.resize(embd[1].shape)
        embd[i] += temp
        i += 1
        # embd.append(map(float, row[1:]))

    print('Loaded GloVe!')
    file.close()
    return vocab,embd


# remove quote punctuations
def remove_quotes(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentences.append(sentence.replace("'","").replace('"',''))
    return new_sentences

# Build vocab dictionary from embedding matrix
def build_vocDict(vocab):
    voc_dict = {}
    for i in range(len(vocab)):
        voc_dict[vocab[i]] = i
    return voc_dict

# ids for words in sentence
def words2ids(sentences, voc_dict, option = 'simple'):
    new_sentences_ids = []
    j = 0
    for sentence in sentences:
        j+=1
        if j % 5000 == 0:
            print ('sentence',str(j))
        sentence_ids = []

        if option == 'nltk':
            sentence = sentence.decode('utf8', 'ignore')
            word_list = tokenize(sentence)
            # print('word_list', word_list)
        elif option == 'simple':
            word_list = sentence.split(" ")
        
        for word in word_list:
            if word.lower() in voc_dict: # Only add word if in dictionary
                word_index = voc_dict[word.lower()]
                sentence_ids.append(word_index)
                
        new_sentences_ids.append(sentence_ids)

    return new_sentences_ids


# embedding vector and ids for words in sentence
def words2ids_vects(sentences, voc_dict, embedding_matrix, option = 'simple'):
    new_sentences_ids = []
    new_sentences_vects = []
    j = 0

    for sentence in sentences:
        j+=1

        if j % 5000 == 0:
            print ('sentence',str(j))

        sentence_ids = []
        sentence_vects = []

        if option == 'nltk':
            word_list = tokenize(sentence)

        elif option == 'simple':
            word_list = sentence.split(" ")
        
        for word in word_list:

            # Only add word if in dictionary
            if word.lower() in voc_dict:
                word_index = voc_dict[word.lower()]
                sentence_ids.append(word_index)
                sentence_vects.append(embedding_matrix[word_index])
                
        new_sentences_ids.append(sentence_ids)
        new_sentences_vects.append(sentence_vects)

    return new_sentences_ids, new_sentences_vects

# tokenize using nltk
def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return map(lambda x:x.encode('utf8', errors = 'ignore'), tokens)


def concatConvert_np(h_list, b_list):
    n_sentences = len(h_list)
    h_b_list = []
    seqlen = []

    for i in range(n_sentences):
        h_b_list.append(h_list[i] + b_list[i])
        seqlen.append(len(h_b_list[i]))
        
    max_len = max(seqlen)
    
    # Convert to numpy with zero padding
    h_b_np = np.zeros((n_sentences, max_len))
    for i in range(n_sentences):
        h_b_np[i,:seqlen[i]] = h_b_list[i]
    
    return h_b_list, h_b_np, np.array(seqlen)

#Convert list to numpy zero padded, 2 distinct matrices for headlines and bodies
def distinctConvert_np(h_list, b_list):
    n_sentences = len(h_list)
    h_seqlen = []
    b_seqlen = []

    for i in range(n_sentences):
        h_seqlen.append(len(h_list[i]))
        b_seqlen.append(len(b_list[i]))
        
    h_max_len = max(h_seqlen)
    b_max_len = max(b_seqlen)

    h_np = np.zeros((n_sentences, h_max_len))
    b_np = np.zeros((n_sentences, b_max_len))
    for i in range(n_sentences):
        h_np[i,:h_seqlen[i]] = h_list[i]
        b_np[i,:b_seqlen[i]] = b_list[i]
        
    return h_np, np.array(h_seqlen), b_np, np.array(b_seqlen)

def save_data_pickle(outfilename, 
                    embedding_type = 'twitter.27B.50d',
                    parserOption = 'nltk'):

    cwd = os.getcwd()

    if embedding_type == 'twitter.27B.50d':
        filename_embeddings = cwd + '/../../glove/glove.twitter.27B.50d.txt'
    else: 
        filename_embeddings = cwd + '/../../glove/glove.6B.50d.txt'

    # GloVe embeddings
    vocab, embd = loadGloVe(filename_embeddings)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])

    # Get vocab dict
    voc_dict = build_vocDict(vocab)
    
    # Read and process data
    h, b, y = read_data(cwd + '/../../') # headline / bodies/ labels
    h_ids, h_vects = words2ids_vects(h, voc_dict, embd, parserOption)
    b_ids, b_vects = words2ids_vects(b, voc_dict, embd, parserOption)
    
    # Concatenated headline_bodies zero padded np matrices; seq. lengths as np vector
    h_b_ids, h_b_np, seqlen = concatConvert_np(h_ids, b_ids)
    h_np, h_seqlen, b_np, b_seqlen = distinctConvert_np(h_ids, b_ids)

    data_dict = {'h_ids':h_ids, 'b_ids':b_ids, 'y':y}
    with open(cwd + outfilename, 'wb') as fp:
        pickle.dump(data_dict, fp)

def get_data(config, 
            filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
            pickle_path = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
            concat = False):

    cwd = os.getcwd()
    filename_embeddings = cwd + filename_embeddings
    
    # GloVe embeddings
    vocab, embd = loadGloVe(filename_embeddings)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd, dtype=np.float64)
    # print("me", embedding)

    # Get vocab dict
    voc_dict = build_vocDict(vocab)

    print('Loading Pickle')
    load_path = cwd + pickle_path
    with open (load_path, 'rb') as fp:
        data_dict = pickle.load(fp)

    h_ids = data_dict['h_ids']
    b_ids = data_dict['b_ids']
    y = data_dict['y']
    print('finished loading Pickle')

    if concat:
        h_b_ids, h_b_np, seqlen = concatConvert_np(h_ids, b_ids)
        output_dict = {'y':y,
                       'h_b_np':h_b_np, 
                       'seqlen':seqlen}
    else:
        h_np, h_seqlen, b_np, b_seqlen = distinctConvert_np(h_ids, b_ids)
        ind_empty = []

        for i in range(np.shape(h_np)[0]):
            if ((h_seqlen[i] == 0) or (b_seqlen[i] == 0)):
                ind_empty.append(i)

        if (len(ind_empty) > 0):
            y = np.delete(y, ind_empty)
            h_np = np.delete(h_np, ind_empty, 0)
            b_np = np.delete(b_np, ind_empty, 0)
            h_seqlen = np.delete(h_seqlen, ind_empty)
            b_seqlen = np.delete(b_seqlen, ind_empty)

        output_dict = {'y':y,
                       'h_np':h_np, 
                       'b_np':b_np, 
                       'h_seqlen':h_seqlen,
                       'b_seqlen':b_seqlen}

    config.embed_size = embedding_dim
    config.pretrained_embeddings = embedding
    config.vocab_size = vocab_size
    return config, output_dict