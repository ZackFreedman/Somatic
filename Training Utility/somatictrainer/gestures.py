import json
import logging
from typing import List, Tuple
import numpy as np
import quaternion

standard_gesture_time = 1000  # Milliseconds
sampling_rate = 10  # Milliseconds

logging.basicConfig(level=logging.INFO)


class Gesture:
    def __init__(self, glyph, raw_data, normalized_quats=None):
        """

        :param normalized_quats:
        :type normalized_quats: list of np.quaternion or None
        :param raw_data:
        :type raw_data list of tuple of np.quaternion, float or None
        :param glyph:
        :type glyph: str
        """
        if normalized_quats is not None and not len(normalized_quats) is 100:
            raise AttributeError('Normalized data invalid - got {} normalized_data instead of {}'
                                 .format(len(normalized_quats), standard_gesture_time / sampling_rate))

        if raw_data is None:
            raise AttributeError('Must provide one source of data')

        if normalized_quats is None:
            normalized_quats = Gesture.normalize_samples(raw_data)

        self.raw_quats = [x[0] for x in raw_data]
        self.raw_timedeltas = [x[1] for x in raw_data]
        self.normalized_data = normalized_quats
        self.glyph = glyph

    def to_dict(self):
        datastore = {
            'g': self.glyph,
            'r': [(s.w, s.x, s.y, s.z, ts) for s, ts in zip(self.raw_quats, self.raw_timedeltas)],
            'n': [(s.w, s.x, s.y, s.z) for s in self.normalized_data]
        }
        return datastore

    @staticmethod
    def from_dict(datastore):
        try:
            glyph = datastore['g']

            raw_data = [(np.quaternion(w=e[0], x=e[1], y=e[2], z=e[3]), e[4]) for e in datastore['r']]
            normalized_quats = [np.quaternion(w=e[0], x=e[1], y=e[2], z=e[3]) for e in datastore['n']]
            assert len(normalized_quats) is round(standard_gesture_time / sampling_rate)

            return Gesture(glyph, raw_data, normalized_quats)

        except (AssertionError, AttributeError, KeyError):
            logging.exception('Error parsing dict {}...'.format(str(datastore)[:20]))

        return None

    @staticmethod
    def normalize_samples(samples: List[Tuple[np.quaternion, int]]):
        if not samples:
            raise AttributeError('Samples and/or timedeltas not provided')

        if not len(samples):
            raise AttributeError('Samples and/or timedeltas list are empty')

        logging.info('Normalizing:\n{0!r}'.format(samples))

        quats = [x[0] for x in samples]
        timedeltas = [x[1] for x in samples]

        scaling_factor = standard_gesture_time / sum(timedeltas)
        # Standardize times to 1 second
        scaled_times = [delta * scaling_factor for delta in timedeltas]

        logging.debug(scaled_times)

        output = []

        # Interpolate to increase/reduce number of samples to required sampling rate
        for earliest_time in range(0, standard_gesture_time - sampling_rate, sampling_rate):
            # For each sample required, find the latest sample before this time, and the earliest sample
            # after this time, and slerp them.

            early_sample = None
            early_time = None
            late_sample = None
            late_time = None

            latest_time = earliest_time + sampling_rate

            sample_time = 0
            for index, sample in enumerate(quats):
                if index:
                    sample_time += scaled_times[index]

                if early_sample is None and sample_time >= earliest_time:
                    # This sample is the latest sample that began earlier than the early time.
                    early_sample = quats[index]
                    early_time = sample_time - scaled_times[index]

                if late_sample is None and sample_time >= latest_time:
                    # This sample is the latest sample that began earlier than the late time.
                    late_sample = quats[index]
                    late_time = sample_time

                if early_sample and late_sample:
                    continue

            if early_sample is None or late_sample is None:
                raise AttributeError('Something went wrong - missing data {0} and {1}. Got {2} and {3}'
                                     .format(earliest_time, latest_time, early_sample, late_sample))

            # amount = (earliest_time - early_time) / (late_time - early_time)  # Just the Arduino map function
            # output.append(quaternion.quaternion_time_series.slerp(early_sample, late_sample, amount))
            output.append(quaternion.quaternion_time_series.slerp(
                early_sample, late_sample, early_time, late_time, earliest_time))

        return output


class GestureTrainingSet:
    big_ole_list_o_glyphs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!@#/ 1234567890'
    short_glyphs = '., '
    current_version = 1  # For deleting old saves

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

            return output

    def save(self, pathspec):
        datastore = {'version': GestureTrainingSet.current_version, 'examples': [x.to_dict() for x in self.examples]}
        with open(pathspec, 'w') as f:
            json.dump(datastore, f)

    def add(self, example: Gesture):
        self.examples.append(example)

    def count(self, glyph):
        return len([example for example in self.examples if example.glyph == glyph])

    def summarize(self):
        return {glyph: self.count(glyph) for glyph in self.big_ole_list_o_glyphs}