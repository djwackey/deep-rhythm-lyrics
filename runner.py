from __future__ import absolute_import
from __future__ import print_function

from generator import LyricsGenerator
import tensorflow as tf


def main(_):
    """main function
    """
    generator = LyricsGenerator(test=True)
    result = generator.run()
    print(result.encode('utf8'))


if __name__ == '__main__':
    tf.app.run()
