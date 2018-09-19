from __future__ import absolute_import
from __future__ import division
import time
import tensorflow as tf
import numpy as np

from util import get_data, save_data_pickle, Progbar, minibatches, pack_labels, split_data, softmax, get_performance, Config

class BaselineLSTM(Config):
    def add_loss_op(self, pred):
        y = tf.reshape(self.labels_placeholder, (-1, )) # Check whether this is necessary
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred, labels = y))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.max_length), name = "x")
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name = "y")
        self.seqlen_placeholder = tf.placeholder(tf.int64, shape=(None), name = "seqlen")
        self.dropout_placeholder = tf.placeholder(tf.float64, name = 'dropout')

    def create_feed_dict(self, inputs_batch, seqlen_batch, labels_batch = None, dropout = 1.0):
        feed_dict = { self.inputs_placeholder: inputs_batch,}

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout

        feed_dict[self.seqlen_placeholder] = seqlen_batch
        return feed_dict

    def add_prediction_op(self):

        if self.config.n_layers <= 1:
            print('layers = ', self.config.n_layers)
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.dropout_placeholder)
            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)
            U = tf.get_variable(name = 'U', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b = tf.get_variable(name = 'b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)

            x = self.add_embedding(option = self.config.trainable_embeddings)
            rnnOutput = tf.nn.dynamic_rnn(cell, inputs = x, dtype = tf.float64, sequence_length = self.seqlen_placeholder)
            finalState = rnnOutput[1][1]
            preds = tf.matmul(finalState, U) + b

        elif self.config.n_layers > 1:
            layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size), output_keep_prob = self.dropout_placeholder) for _ in range(self.config.n_layers)]
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(layers)

            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)
            U = tf.get_variable(name = 'U', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b = tf.get_variable(name = 'b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            x = self.add_embedding(option = self.config.trainable_embeddings)
            rnnOutput = tf.nn.dynamic_rnn(stacked_lstm, inputs = x, dtype = tf.float64, sequence_length = self.seqlen_placeholder)
            print('layers = ', self.config.n_layers)
            finalState = rnnOutput[1][self.config.n_layers - 1][1]
            preds = tf.matmul(finalState, U) + b
        return preds

    # maps input tokens to vectors using embedding lookup
    def add_embedding(self, option = 'Constant'):
        if option == 'Variable':
            embeddings_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.inputs_placeholder)
        elif option == 'Constant':
            embeddings_temp = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.inputs_placeholder)

        embeddings = tf.reshape(embeddings_temp, shape = (-1, self.config.max_length, self.config.embed_size))
        return embeddings

    def train_on_batch(self, sess, inputs_batch, labels_batch, seqlen_batch):
        labels_batch = np.reshape(labels_batch, (-1, 1))
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, seqlen_batch = seqlen_batch, dropout = self.config.dropout) # MODIF
        print(inputs_batch.shape)
        print(len(labels_batch))
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch, seqlen_batch):
        feed = self.create_feed_dict(inputs_batch, seqlen_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []

        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            losses.append(loss)
            prog.update(i + 1, [("train loss", loss)])

        return losses

    def fit(self, sess, train, dev_data_np, dev_seqlen, dev_labels):
        losses_epochs = []
        dev_performances_epochs = []
        dev_predictions_epochs = []
        dev_predicted_classes_epochs = []
        for epoch in range(self.config.n_epochs):
            print("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss = self.run_epoch(sess, train)

            # Computing predictions
            dev_predictions = self.predict_on_batch(sess, dev_data_np, dev_seqlen)

            # Computing development performance
            dev_predictions = softmax(np.array(dev_predictions))
            dev_predicted_classes = np.argmax(dev_predictions, axis = 1)
            dev_performance = get_performance(dev_predicted_classes, dev_labels, n_classes = 4)

            # Adding to global outputs
            dev_predictions_epochs.append(dev_predictions)
            dev_predicted_classes_epochs.append(dev_predicted_classes)
            dev_performances_epochs.append(dev_performance)
            losses_epochs.append(loss)

        return losses_epochs, dev_performances_epochs, dev_predicted_classes_epochs, dev_predictions_epochs

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.seqlen_placeholder = None
        self.dropout_placeholder = None
        self.build()

def run_save_data_pickle():
    save_data_pickle(outfilename = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                    embedding_type = 'twitter.27B.50d',
                    parserOption = 'simple')
    
def run_lstm(config, final = False):
    config, data_dict = get_data(config, filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
                                pickle_path = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                                concat = True)

    y = data_dict['y']
    h_b_np = data_dict['h_b_np']
    seqlen = data_dict['seqlen']

    if config.max_length is not None:
        max_length = config.max_length
        if np.shape(h_b_np)[1] > max_length:
            h_b_np = h_b_np[:, 0:max_length]
        seqlen = np.minimum(seqlen, max_length)

    # Set maximum dataset size for testing purposes
    data = pack_labels(h_b_np, y, seqlen)
    if config.num_samples is not None:
        num_samples = config.num_samples
        data = data[0:num_samples - 1]

    # Split data, result is still packed
    train_data, dev_data, test_data, train_indices, dev_indices, test_indices = split_data(data, prop_train = 0.6, prop_dev = 0.2, seed = 56)

    # Dev
    dev_labels = y[dev_indices]
    dev_data_np = h_b_np[dev_indices, :]
    dev_seqlen = seqlen[dev_indices]

    # Test
    test_labels = y[test_indices]
    test_data_np = h_b_np[test_indices, :]
    test_seqlen = seqlen[test_indices]

    ## Config determined at data loading:
    config.num_samples = len(train_indices)
    config.max_length = np.shape(h_b_np)[1]

    # For final test combine test and dev - Reassign test to dev
    if final:
        train_dev_indices = train_indices + dev_indices
        train_data = [data[i] for i in train_dev_indices]
        dev_data_np = test_data_np
        dev_seqlen = test_seqlen
        dev_labels = test_labels
        config.num_samples = len(train_dev_indices)

    with tf.Graph().as_default():
        tf.set_random_seed(59)
        print("Building model...",)
        start = time.time()
        model = BaselineLSTM(config)
        print("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)
            losses_ep, dev_performances_ep, dev_predicted_classes_ep, dev_predictions_ep = model.fit(session, train_data, dev_data_np, dev_seqlen, dev_labels)

    print('Dev Performance ', dev_performances_ep)
    return losses_ep, dev_predicted_classes_ep, dev_performances_ep
