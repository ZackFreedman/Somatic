import json
import logging
from typing import List, Tuple, Any
import numpy as np

standard_gesture_length = 50  # Length of (yaw, pitch) coordinates per gesture to be fed into ML algorithm

logging.basicConfig(level=logging.INFO)


class Gesture:
    def __init__(self, glyph, bearings, raw_data):
        """

        :param bearings: Standardized ordered list of (yaw/pitch) coordinates
        :type bearings: np.array
        :param raw_data: Raw data collected during training,
        in case we make a processing boo-boo and need to retroactively fix things
        :type raw_data: list
        :param glyph: Which letter or opcode this gesture represents
        :type glyph: str
        """

        if bearings.shape != (50, 2):
            raise AttributeError('Data invalid - got {} orientations instead of {}'
                                 .format(len(bearings), standard_gesture_length))

        self.bearings = bearings
        self.raw_data = raw_data
        self.glyph = glyph

    def to_dict(self):
        datastore = {
            'g': self.glyph,
            'b': self.bearings.tolist(),
            'r': self.raw_data
        }
        return datastore

    @staticmethod
    def from_dict(datastore):
        try:
            glyph = datastore['g']

            bearings = np.array(datastore['b'])
            assert len(bearings) == standard_gesture_length

            raw_data = datastore['r']

            return Gesture(glyph, bearings, raw_data)

        except (AssertionError, AttributeError, KeyError):
            logging.exception('Error parsing dict {}...'.format(str(datastore)[:20]))

        return None


class GestureTrainingSet:
    examples: List[Gesture]

    big_ole_list_o_glyphs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!@#/ 1234567890'
    short_glyphs = '., '
    current_version = 3  # For deleting old saves

    def __init__(self):
        self.target_examples_per_glyph = 100

        self.examples = []

    @staticmethod
    def load(pathspec):
        with open(pathspec, 'r') as f:
            datastore = json.load(f)

            if 'version' not in datastore or datastore['version'] != GestureTrainingSet.current_version:
                logging.warning("Saved file is outdated, not loading")
                return

            output = GestureTrainingSet()

            for sample_record in datastore['examples']:
                output.add(Gesture.from_dict(sample_record))

            logging.info(output)

            return output

    def save(self, pathspec):
        datastore = {'version': GestureTrainingSet.current_version, 'examples': [x.to_dict() for x in self.examples]}
        with open(pathspec, 'w') as f:
            json.dump(datastore, f)

    def add(self, example: Gesture):
        self.examples.append(example)

    def get_examples_for(self, glyph):
        return [example for example in self.examples if example.glyph == glyph]

    def count(self, glyph):
        return len(self.get_examples_for(glyph))

    def summarize(self):
        return {glyph: self.count(glyph) for glyph in self.big_ole_list_o_glyphs}

    def remove(self, example):
        if example in self.examples:
            self.examples.remove(example)

    def move(self, example, new_glyph):
        if example in self.examples:
            example.glyph = new_glyph

    def remove_at(self, glyph, index):
        if index < self.count(glyph):
            self.examples.remove(self.get_examples_for(glyph)[index])

    def to_training_set(self):
        data = []
        labels = []

        for example in self.examples:
            data.append(example.bearings)
            labels.append(ord(example.glyph) - ord('a'))

        return np.array(data), np.array(labels)
