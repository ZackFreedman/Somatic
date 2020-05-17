import logging
import os
import time
from typing import List
import numpy as np
import uuid
import pickle

standard_gesture_length = 50  # Length of (yaw, pitch) coordinates per gesture to be fed into ML algorithm

_log_level = logging.DEBUG


class Gesture:
    logger = logging.getLogger(__name__)
    logger.setLevel(_log_level)
    
    def __init__(self, glyph, bearings, raw_data, gesture_uuid=None):
        """

        :param bearings: Standardized ordered list of (yaw/pitch) coordinates
        :type bearings: np.array
        :param raw_data: Raw data collected during training,
        in case we make a processing boo-boo and need to retroactively fix things
        :type raw_data: list
        :param glyph: Which letter or opcode this gesture represents
        :type glyph: str
        :param gesture_uuid: A unique identifier used to tie the gesture to UI elements
        :type gesture_uuid: uuid.UUID
        """

        if bearings.shape != (50, 2):
            raise AttributeError('Data invalid - got {} orientations instead of {}'
                                 .format(len(bearings), standard_gesture_length))
        self.bearings = bearings
        self.raw_data = raw_data
        self.glyph = glyph

        if gesture_uuid is not None:
            self.uuid = gesture_uuid
        else:
            self.uuid = uuid.uuid4()

    def to_dict(self):
        datastore = {
            'g': self.glyph,
            'b': self.bearings.tolist(),
            'r': self.raw_data,
            'id': str(self.uuid)
        }
        return datastore

    @staticmethod
    def from_dict(datastore):
        try:
            if 'id' in datastore:
                gesture_uuid = uuid.UUID(datastore['id'])
            else:
                gesture_uuid = None

            glyph = datastore['g']

            bearings = np.array(datastore['b'])
            assert len(bearings) == standard_gesture_length

            raw_data = datastore['r']

            return Gesture(glyph, bearings, raw_data, gesture_uuid)

        except (AssertionError, AttributeError, KeyError):
            Gesture.logger.exception('Gesture class: Error parsing dict {}...'.format(str(datastore)[:20]))

        return None


class GestureTrainingSet:
    examples: List[Gesture]

    big_ole_list_o_glyphs = '\x08\n !"#$\',-./0123456789?@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    short_glyphs = '., \x08l-/'
    current_version = 3  # For deleting old saves

    logger = logging.getLogger(__name__)
    logger.setLevel(_log_level)

    def __init__(self):
        self.target_examples_per_glyph = 100
        self.examples = []
        # self.unidentified_examples = []

    @staticmethod
    def load(pathspec):
        with open(pathspec, 'rb') as f:
            output = GestureTrainingSet()

            output.examples = pickle.load(f)

            GestureTrainingSet.logger.debug('GestureTrainingSet class: Loaded {}'.format(output))

            return output

    def save(self, pathspec):
        t = time.perf_counter()
        GestureTrainingSet.logger.debug('Generating save dict took {}'.format(time.perf_counter() - t))

        t = time.perf_counter()
        # Save unidentified samples here?
        with open(pathspec + '.tmp', 'wb') as f:
            pickle.dump(self.examples, f)

        GestureTrainingSet.logger.debug('Saving took {}'.format(time.perf_counter() - t))

        if os.path.exists(pathspec):
            os.remove(pathspec)
        os.rename(pathspec + '.tmp', pathspec)

    def add(self, example: Gesture):
        self.examples.append(example)

    def get_examples_for(self, glyph):
        return [example for example in self.examples if example.glyph == glyph]

    def count(self, glyph):
        return len(self.get_examples_for(glyph))

    def summarize(self):
        return {glyph: self.count(glyph) for glyph in self.big_ole_list_o_glyphs}

    def remove(self, example_or_uuid):
        if type(example_or_uuid) is Gesture:
            example = example_or_uuid
            if example in self.examples:
                self.examples.remove(example)

        elif type(example_or_uuid) is uuid.UUID:
            gesture_uuid = example_or_uuid
            for example in self.examples:
                if gesture_uuid == example.uuid:
                    self.examples.remove(example)
                    break

    def move(self, example, new_glyph):
        if example in self.examples:
            example.glyph = new_glyph

    def remove_at(self, glyph, index):
        if index < self.count(glyph):
            self.examples.remove(self.get_examples_for(glyph)[index])

    def get_character_map(self, type='decoding'):
        chars = np.unique([x.glyph for x in self.examples])
        if type is 'decoding':
            return {i: chars[i] for i in range(len(chars))}
        elif type is 'encoding':
            return {chars[i]: i for i in range(len(chars))}
        else:
            raise AttributeError('Char map type must be "encoding" or "decoding"')

    def to_training_set(self):
        char_map = self.get_character_map(type='encoding')

        data = []
        labels = []

        for example in self.examples:
            data.append(example.bearings)
            labels.append(char_map[example.glyph])

        return np.array(data), np.array(labels)
