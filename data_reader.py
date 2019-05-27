import json
import os
import random

import numpy as np


class DataReader(object):
    def __init__(self):
        self.lyrics = []
        self.lyrics_path = '/data/chinese-hiphop-lyrics/'
        self.lyric_indices = []
        self.vocab_lookup = {}
        self.index_lookup = {}

    def get_path(self):
        """
        @return: The path to the specified artist's lyric data.
        """
        return self.lyrics_path

    def load_lyrics(self):
        """
        Read lyrics from file into self.lyrics - a 2D list of dimensions [songs, song_words].
        """
        path = self.get_path()
        letters = [chr(letter) for letter in range(ord('A'), ord('Z') + 1)]
        for letter in letters:
            lrc_path = os.path.join(path, letter)
            for filename in os.listdir(lrc_path):
                with open(os.path.join(lrc_path, filename), 'r') as f:
                    try:
                        print filename
                        artist = json.loads(f.read())
                        for song in artist['songs']:
                            song_lyrics = []
                            for sentence in song['lyrics']:
                                # lyrics += sentence.encode('utf8')
                                song_lyrics.append(sentence.strip())
                            self.lyrics.append(song_lyrics)
                    except Exception as e:
                        print e
        # print lyrics
        # self.lyrics = sorted(list(set(self.lyrics)))
        # data_size, _vocab_size = len(lyrics), len(self.lyrics)
        # print("data has {0} characters, {1} unique.".format(
        #     data_size, _vocab_size))

    def get_vocab(self):
        """
        @return: An array of unique words (tokens) with the bottom THRESHOLD_COUNT least
        frequent words converted to '*UNK*'
        """
        # Load the lyric data if it hasn't been loaded already
        if len(self.lyrics) == 0:
            self.load_lyrics()

        # Collapses the 2D array to a 1D array of words
        all_words = reduce(lambda a, b: a + b, self.lyrics)

        # TODO: Find out why this UNK code causes differences between Linux and OS X vocabularies
        # # convert THRESHOLD_COUNT frequent words to '*UNK*'
        # THRESHOLD_COUNT = 10
        # least_referenced = Counter(all_words).most_common()[:-(THRESHOLD_COUNT + 1):-1]
        # least_referenced = [tup[0] for tup in least_referenced] # grab word from (word, count) tuple
        # print least_referenced
        #
        # self.lyrics = [map(lambda word: c.UNK if word in least_referenced else word, song)
        #                for song in self.lyrics]
        # # reset all_words to include UNKs
        # all_words = reduce(lambda a, b: a + b, self.lyrics)

        # get a sorted list of unique word tokens
        tokens = sorted(list(set(all_words)))

        # creates a map from word to index
        self.vocab_lookup = dict((word, i) for i, word in enumerate(tokens))
        # Converts words in self.lyrics to the appropriate indices.
        self.lyric_indices = [
            map(lambda word: self.vocab_lookup[word], song)
            for song in self.lyrics
        ]

        print len(tokens)

        return tokens

    def get_train_batch(self, batch_size, seq_len):
        """
        Gets a batch of sequences for training.
        @param batch_size: The number of sequences in the batch.
        @param seq_len: The number of words in a sequence.
        @return: A tuple of arrays of shape [batch_size, seq_len].
        """
        inputs = np.empty([batch_size, seq_len], dtype=int)
        targets = np.empty([batch_size, seq_len], dtype=int)

        for i in xrange(batch_size):
            inp, target = self.get_seq(seq_len)
            inputs[i] = inp
            targets[i] = target

        return inputs, targets

    def get_seq(self, seq_len):
        """
        Gets a single pair of sequences (input, target) from a random song.
        @param seq_len: The number of words in a sequence.
        @return: A tuple of sequences, (input, target) offset from each other by one word.
        """
        # Pick a random song. Must be longer than seq_len
        for i in xrange(1000):  # cap at 1000 tries
            song = random.choice(self.lyric_indices)
            if len(song) > seq_len:
                break

        # Take a sequence of (seq_len) from the song lyrics
        i = random.randint(0, len(song) - (seq_len + 1))
        inp = np.array(song[i:i + seq_len], dtype=int)
        target = np.array(song[i + 1:i + seq_len + 1], dtype=int)
        return inp, target
