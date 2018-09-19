import tensorflow as tf
import numpy as np
import random
from util import split_indices, softmax, get_performance, save_data_pickle, get_data, Config

class BOWModel(Config):
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
        self.headings_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.h_max_len), name="headings")
        self.bodies_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.b_max_len), name="bodies")
        self.headings_lengths_placeholder = tf.placeholder(tf.float64, shape=(None), name="headings_lengths")
        self.bodies_lengths_placeholder = tf.placeholder(tf.float64, shape=(None), name="bodies_lengths")
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name="labels")

    def create_feed_dict(self, headings_batch, bodies_batch, headings_lengths_batch, bodies_lengths_batch,
                         labels_batch=None):
        feed_dict = {
            self.headings_placeholder: headings_batch,
            self.bodies_placeholder: bodies_batch,
            self.headings_lengths_placeholder: headings_lengths_batch,
            self.bodies_lengths_placeholder: bodies_lengths_batch,
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self, option='Constant'):
        embeddings_headings_temp = tf.nn.embedding_lookup(params=self.config.pretrained_embeddings, ids=self.headings_placeholder)
        embeddings_bodies_temp = tf.nn.embedding_lookup(params=self.config.pretrained_embeddings, ids=self.bodies_placeholder)

        embeddings_headings = tf.reshape(embeddings_headings_temp, shape=(-1, self.config.h_max_len, self.config.embed_size))
        embeddings_bodies = tf.reshape(embeddings_bodies_temp, shape=(-1, self.config.b_max_len, self.config.embed_size))

        return embeddings_headings, embeddings_bodies

    def add_bow_input(self):
        headings, bodies = self.add_embedding(option=self.config.trainable_embeddings)
        # isall_zero = tf.equal(self.headings_lengths_placeholder, 0)

        # averaging operation for bodies n headings
        headings_bag = tf.divide(tf.reduce_sum(headings, axis=1), tf.reshape(self.headings_lengths_placeholder, shape=(-1, 1)))
        bodies_bag = tf.divide(tf.reduce_sum(bodies, axis=1), tf.reshape(self.bodies_lengths_placeholder, shape=(-1, 1)))

        x = tf.concat(axis=1, values=[headings_bag, bodies_bag])
        return x

    def add_prediction_op(self):
        hidden_size_2 = np.floor(self.config.hidden_next ** 2 * self.config.hidden_size)
        hidden_size_3 = np.floor(self.config.hidden_next ** 3 * self.config.hidden_size)
        hidden_size_4 = np.floor(self.config.hidden_next ** 4 * self.config.hidden_size)
        hidden_size_5 = np.floor(self.config.hidden_next ** 5 * self.config.hidden_size)

        x = self.add_bow_input()
        theInitializer = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float64)
        if not self.config.n_layers:
            W = tf.get_variable(name='W', shape=(2 * self.config.embed_size, self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            # bias
            c = tf.get_variable(name='c', shape=(self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            pred = tf.matmul(x, W) + c

        elif self.config.n_layers == 1:
            U0 = tf.get_variable(name='U0', shape=(2 * self.config.embed_size, self.config.hidden_size), initializer=theInitializer, dtype=tf.float64)
            c0 = tf.get_variable(name='c0', shape=(self.config.hidden_size), initializer=theInitializer, dtype=tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0)
            U1 = tf.get_variable(name='U1', shape=(self.config.hidden_size, self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            c1 = tf.get_variable(name='c1', shape=(self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            pred = tf.matmul(h1, U1) + c1

        elif self.config.n_layers == 2:
            U0 = tf.get_variable(name='U0', shape=(2 * self.config.embed_size, self.config.hidden_size), initializer=theInitializer, dtype=tf.float64)
            c0 = tf.get_variable(name='c0', shape=(self.config.hidden_size), initializer=theInitializer, dtype=tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0)
            U1 = tf.get_variable(name='U1', shape=(self.config.hidden_size, hidden_size_2), initializer=theInitializer, dtype=tf.float64)
            c1 = tf.get_variable(name='c1', shape=(hidden_size_2), initializer=theInitializer, dtype=tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1)
            U2 = tf.get_variable(name='U2', shape=(hidden_size_2, self.config.n_classes), initializer=theInitializer,
                                 dtype=tf.float64)
            c2 = tf.get_variable(name='c2', shape=(self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            pred = tf.matmul(h2, U2) + c2

        elif self.config.n_layers == 3:
            U0 = tf.get_variable(name='U0', shape=(2 * self.config.embed_size, self.config.hidden_size),
                                 initializer=theInitializer, dtype=tf.float64)
            c0 = tf.get_variable(name='c0', shape=(self.config.hidden_size), initializer=theInitializer,
                                 dtype=tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0)
            U1 = tf.get_variable(name='U1', shape=(self.config.hidden_size, hidden_size_2), initializer=theInitializer,
                                 dtype=tf.float64)
            c1 = tf.get_variable(name='c1', shape=(hidden_size_2), initializer=theInitializer, dtype=tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1)
            U2 = tf.get_variable(name='U2', shape=(hidden_size_2, hidden_size_3), initializer=theInitializer,
                                 dtype=tf.float64)
            c2 = tf.get_variable(name='c2', shape=(hidden_size_3), initializer=theInitializer, dtype=tf.float64)
            h3 = tf.nn.relu(tf.matmul(h2, U2) + c2)
            U3 = tf.get_variable(name='U3', shape=(hidden_size_3, self.config.n_classes), initializer=theInitializer,
                                 dtype=tf.float64)
            c3 = tf.get_variable(name='c3', shape=(self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            pred = tf.matmul(h3, U3) + c3

        elif self.config.n_layers == 4:
            U0 = tf.get_variable(name='U0', shape=(2 * self.config.embed_size, self.config.hidden_size),
                                 initializer=theInitializer, dtype=tf.float64)
            c0 = tf.get_variable(name='c0', shape=(self.config.hidden_size), initializer=theInitializer,
                                 dtype=tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0)  # batch_size, hidden_size
            U1 = tf.get_variable(name='U1', shape=(self.config.hidden_size, hidden_size_2), initializer=theInitializer,
                                 dtype=tf.float64)
            c1 = tf.get_variable(name='c1', shape=(hidden_size_2), initializer=theInitializer, dtype=tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1)  # batch_size, hidden_size_2
            U2 = tf.get_variable(name='U2', shape=(hidden_size_2, hidden_size_3), initializer=theInitializer,
                                 dtype=tf.float64)
            c2 = tf.get_variable(name='c2', shape=(hidden_size_3), initializer=theInitializer, dtype=tf.float64)
            h3 = tf.nn.relu(tf.matmul(h2, U2) + c2)  # batch_size, hidden_size_3
            U3 = tf.get_variable(name='U3', shape=(hidden_size_3, hidden_size_4), initializer=theInitializer,
                                 dtype=tf.float64)
            c3 = tf.get_variable(name='c3', shape=(hidden_size_4), initializer=theInitializer, dtype=tf.float64)
            h4 = tf.nn.relu(tf.matmul(h3, U3) + c3)  # batch_size, hidden_size_4
            U4 = tf.get_variable(name='U4', shape=(hidden_size_4, self.config.n_classes), initializer=theInitializer,
                                 dtype=tf.float64)
            c4 = tf.get_variable(name='c4', shape=(self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            pred = tf.matmul(h4, U4) + c4

        elif self.config.n_layers == 5:
            U0 = tf.get_variable(name='U0', shape=(2 * self.config.embed_size, self.config.hidden_size),
                                 initializer=theInitializer, dtype=tf.float64)
            c0 = tf.get_variable(name='c0', shape=(self.config.hidden_size), initializer=theInitializer,
                                 dtype=tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0)
            U1 = tf.get_variable(name='U1', shape=(self.config.hidden_size, hidden_size_2), initializer=theInitializer,
                                 dtype=tf.float64)
            c1 = tf.get_variable(name='c1', shape=(hidden_size_2), initializer=theInitializer, dtype=tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1)
            U2 = tf.get_variable(name='U2', shape=(hidden_size_2, hidden_size_3), initializer=theInitializer,
                                 dtype=tf.float64)
            c2 = tf.get_variable(name='c2', shape=(hidden_size_3), initializer=theInitializer, dtype=tf.float64)
            h3 = tf.nn.relu(tf.matmul(h2, U2) + c2)
            U3 = tf.get_variable(name='U3', shape=(hidden_size_3, hidden_size_4), initializer=theInitializer,
                                 dtype=tf.float64)
            c3 = tf.get_variable(name='c3', shape=(hidden_size_4), initializer=theInitializer, dtype=tf.float64)
            h4 = tf.nn.relu(tf.matmul(h3, U3) + c3)
            U4 = tf.get_variable(name='U4', shape=(hidden_size_4, hidden_size_5), initializer=theInitializer,
                                 dtype=tf.float64)
            c4 = tf.get_variable(name='c4', shape=(hidden_size_5), initializer=theInitializer, dtype=tf.float64)
            h5 = tf.nn.relu(tf.matmul(h4, U4) + c4)
            U5 = tf.get_variable(name='U5', shape=(hidden_size_5, self.config.n_classes), initializer=theInitializer,
                                 dtype=tf.float64)
            c5 = tf.get_variable(name='c5', shape=(self.config.n_classes), initializer=theInitializer, dtype=tf.float64)
            pred = tf.matmul(h5, U5) + c5

        return pred

    def train_on_batch(self, sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch):
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)

        # if (np.isnan(loss)):
        #     print('headings', h_batch)
        #     print('bodies', b_batch)
        #     print('nh_len', h_len_batch)
        #     print('b_len', b_len_batch)
        #     print('labels', y_batch)
        #     assert (False)

        return loss

    def predict_on_batch(self, sess, h_batch, b_batch, h_len_batch, b_len_batch):
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, h_np, b_np, h_len, b_len, y):
        losses = []
        ind = list(range(self.config.num_samples))
        random.shuffle(ind)

        batch_start = 0
        batch_end = 0
        N = self.config.batch_size
        num_batches = self.config.num_samples / N

        # run batches
        for i in range(int(num_batches)):
            batch_start = (i * N)
            batch_end = (i + 1) * N
            indices = ind[batch_start:batch_end]
            h_batch = h_np[indices, :]
            b_batch = b_np[indices, :]
            h_len_batch = h_len[indices]
            b_len_batch = b_len[indices]
            y_batch = y[indices]
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
            losses.append(loss)

            if (i % (1 + num_batches / 10)) == 0:
                print('batch: ', i, ', loss: ', loss)

        if (batch_end < self.config.num_samples):
            indices = ind[batch_end:]
            h_batch = h_np[indices, :]
            b_batch = b_np[indices, :]
            h_len_batch = h_len[indices]
            b_len_batch = b_len[indices]
            y_batch = y[indices]
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
            losses.append(loss)
            print('batch: ', i, ', loss: ', loss)

        return losses


    def fit(self, sess, h_np, b_np, h_len, b_len, y, dev_h, dev_b, dev_h_len, dev_b_len, dev_y):
        losses_epochs = []
        dev_performances_epochs = []
        dev_predictions_epochs = []
        dev_predicted_classes_epochs = []

        for epoch in range(self.config.n_epochs):
            print('-------new epoch---------')
            loss = self.run_epoch(sess, h_np, b_np, h_len, b_len, y)

            # Computing predictions
            dev_predictions = self.predict_on_batch(sess, dev_h, dev_b, dev_h_len, dev_b_len)

            # Computing development performance
            dev_predictions = softmax(np.array(dev_predictions))
            dev_predicted_classes = np.argmax(dev_predictions, axis=1)
            dev_performance = get_performance(dev_predicted_classes, dev_y, n_classes=4)

            # Adding to global outputs
            dev_predictions_epochs.append(dev_predictions)
            dev_predicted_classes_epochs.append(dev_predicted_classes)
            dev_performances_epochs.append(dev_performance)
            losses_epochs.append(loss)

            print('EPOCH: ', epoch, ', LOSS: ', np.mean(loss))

        return losses_epochs, dev_performances_epochs, dev_predicted_classes_epochs, dev_predictions_epochs


    def __init__(self, config):
        self.config = config
        self.headings_placeholder = None
        self.bodies_placeholder = None
        self.headings_lengths_placeholder = None
        self.bodies_lengths_placeholder = None
        self.labels_placeholder = None
        self.build()

def run_save_data_pickle():
    save_data_pickle(outfilename = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                     embedding_type = 'twitter.27B.50d', parserOption = 'simple')

def run_bow(config, split = True, final = False): #M

    config, data_dict = get_data(config, 
            filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
            pickle_path = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
            concat = False)

    y = data_dict['y']
    h = data_dict['h_np']
    b = data_dict['b_np']
    h_len = data_dict['h_seqlen']
    b_len = data_dict['b_seqlen']

    # Do shortening of dataset ## affects number of samples and max_len.
    if config.num_samples is not None:
        np.random.seed(1)
        ind = range(np.shape(h)[0])
        random.shuffle(ind)
        indices = ind[0:config.num_samples]
        h = h[indices,:]
        b = b[indices,:]
        h_len = h_len[indices]
        b_len = b_len[indices]
        y = y[indices]

    if config.h_max_len is not None:
        h_max_len = config.h_max_len
        if np.shape(h)[1] > h_max_len:
            h = h[:, 0:h_max_len]
        h_len = np.minimum(h_len, h_max_len)

    if config.b_max_len is not None:
        b_max_len = config.b_max_len
        if np.shape(b)[1] > b_max_len:
            b = b[:, 0:b_max_len]
        b_len = np.minimum(b_len, b_max_len)

    if split:
        # Split data
        train_indices, dev_indices, test_indices = split_indices(np.shape(h)[0])
        # Divide data
        train_h = h[train_indices,:]
        train_b = b[train_indices,:]
        train_h_len = h_len[train_indices]
        train_b_len = b_len[train_indices]
        train_y = y[train_indices]

        # Development
        dev_h = h[dev_indices,:]
        dev_b = b[dev_indices,:]
        dev_h_len = h_len[dev_indices]
        dev_b_len = b_len[dev_indices]
        dev_y = y[dev_indices]

        if final:
            # Combine train and dev
            train_dev_indices = list(train_indices) + list(dev_indices)
            train_h = h[train_dev_indices,:]
            train_b = b[train_dev_indices,:]
            train_h_len = h_len[train_dev_indices]
            train_b_len = b_len[train_dev_indices]
            train_y = y[train_dev_indices]

            # Set dev to test
            dev_h = h[test_indices,:]
            dev_b = b[test_indices,:]
            dev_h_len = h_len[test_indices]
            dev_b_len = b_len[test_indices]
            dev_y = y[test_indices]

    ## Passing parameter_dict to config settings
    ## Changes to config  based on data shape
    assert(np.shape(train_h)[0] == np.shape(train_b)[0] == np.shape(train_y)[0] == np.shape(train_h_len)[0] == np.shape(train_b_len)[0])
    config.num_samples = np.shape(train_h)[0]
    config.h_max_len = np.shape(train_h)[1]
    config.b_max_len = np.shape(train_b)[1]

    print('Starting TensorFlow operations')
    print ('With hidden layers: ', config.n_layers)

    with tf.Graph().as_default():
        tf.set_random_seed(1)
        print (config)
        model = BOWModel(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            losses_ep, dev_performances_ep, dev_predicted_classes_ep, dev_predictions_ep = model.fit(session, train_h, train_b, train_h_len, train_b_len, train_y, dev_h, dev_b, dev_h_len, dev_b_len, dev_y)

    print('Losses ', losses_ep)
    print('Dev Performance ', dev_performances_ep)
    return losses_ep, dev_predicted_classes_ep, dev_performances_ep
