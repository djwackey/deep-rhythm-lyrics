from __future__ import absolute_import
from __future__ import print_function

import constants as c
from data_reader import DataReader
from model import Model
import tensorflow as tf


class LyricsGenerator():
    def __init__(self, model_name='model', test=False):
        self.session = tf.Session()
        print('Process data...')
        self.data_reader = DataReader()
        self.vocab = self.data_reader.get_vocab()
        print('Init model...')
        self.model = Model(self.session, self.vocab, c.BATCH_SIZE, c.SEQ_LEN,
                           c.CELL_SIZE, c.NUM_LAYERS, test)
        print('Init variables...')
        self.test = test
        self.saver = tf.train.Saver(max_to_keep=None)
        self.session.run(tf.global_variables_initializer())
        self.model_name = model_name

    def train(self):
        """Runs a training loop on the model.
        """
        while True:
            inputs, targets = self.data_reader.get_train_batch(
                c.BATCH_SIZE, c.SEQ_LEN)
            print('Training model...')

            feed_dict = {
                self.model.inputs: inputs,
                self.model.targets: targets
            }
            global_step, loss, _ = self.session.run(
                [self.model.global_step, self.model.loss, self.model.train_op],
                feed_dict=feed_dict)

            print('Step: %d | loss: %f' % (global_step, loss))
            if global_step % c.MODEL_SAVE_FREQ == 0:
                print('Saving model...')
                model_path = '{}{}.ckpt'.format(c.MODEL_SAVE_DIR,
                                                self.model_name)
                self.saver.save(self.session,
                                model_path,
                                global_step=global_step)

    def generate(self):
        """Generate the lyrics
        """
        return self.model.generate()

    def run(self):
        # if self.test and self._load_saved_model():
        if self.test:
            return self.generate()
        else:
            return self.train()

    def _load_saved_model(self):
        print("model loading ...")
        ok = True
        try:
            model_path = '{}{}.ckpt-{}'.format(c.MODEL_SAVE_DIR,
                                               self.model_name,
                                               c.MODEL_SAVE_FREQ)
            self.saver.restore(self.session, model_path)
        except ValueError:
            ok = False
        print("Done!")
        return ok
