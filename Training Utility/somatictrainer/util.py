import logging
import time

import numpy as np
import quaternion

logging.basicConfig(level=logging.INFO)


def lookRotation(forward, up):
    """
    Quaternion that rotates world to face Forward, while keeping orientation dictated by Up
    See https://answers.unity.com/questions/467614/what-is-the-source-code-of-quaternionlookrotation.html

    :type forward: np.array
    :type up: np.array
    """

    up /= np.linalg.norm(up)

    vector = forward / np.linalg.norm(forward)

    vector2 = np.cross(up, vector)
    vector2 /= np.linalg.norm(vector2)

    vector3 = np.cross(vector, vector2)

    m00 = vector2[0]
    m01 = vector2[1]
    m02 = vector2[2]
    m10 = vector3[0]
    m11 = vector3[1]
    m12 = vector3[2]
    m20 = vector[0]
    m21 = vector[1]
    m22 = vector[2]

    num8 = (m00 + m11) + m22

    output = quaternion.quaternion()

    if num8 > 0:
        num = np.sqrt(num8 + 1)

        output.w = num / 2

        num = 0.5 / num

        output.x = (m12 - m21) * num
        output.y = (m20 - m02) * num
        output.z = (m01 - m10) * num

    elif m00 >= m11 and m00 >= m22:
        num7 = np.sqrt((m00 + 1) - m11 - m22)
        num4 = 0.5 / num7

        output.x = num7 / 2
        output.y = (m01 + m10) * num4
        output.z = (m02 + m20) * num4
        output.w = (m12 - m21) * num4

    elif m11 > m22:
        num6 = np.sqrt(m11 + 1 - m00 - m22)
        num3 = 0.5 / num6

        output.x = (m10 + m01) * num3
        output.y = num6 / 2
        output.z = (m21 + m12) * num3
        output.w = (m20 - m02) * num3

    else:
        num5 = np.sqrt(m22 + 1 - m00 - m11)
        num2 = 0.5 / num5

        output.x = (m20 + m02) * num2
        output.y = (m21 + m12) * num2
        output.z = num5 / 2
        output.w = (m01 - m10) * num2

    return output


def custom_interpolate(value, in_min, in_max, out_min, out_max, clamp=False):
    interpolated = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    if clamp:
        return np.clip(interpolated, out_min, out_max)
    else:
        return interpolated


def custom_euler(q):
    # h = np.arctan2(np.square(q.x) - np.square(q.y) - np.square(q.z) + np.square(q.w), 2 * (q.x * q.y + q.z * q.w))
    # h = np.arctan2(2 * (q.x * q.y + q.z * q.w), np.square(q.x) - np.square(q.y) - np.square(q.z) + np.square(q.w))
    # p = np.arcsin(np.clip(-2 * (q.x * q.z - q.y * q.w), -1, 1))
    # r = np.arctan2(np.square(q.z) + np.square(q.w) - np.square(q.x) - np.square(q.y), 2 * (q.x * q.w + q.y * q.z))
    # r = np.arctan2(2 * (q.x * q.w + q.y * q.z), np.square(q.z) + np.square(q.w) - np.square(q.x) - np.square(q.y))

    h = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (np.square(q.y) + np.square(q.z)))
    p = np.arcsin(2 * (q.w * q.y - q.z * q.x))
    r = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (np.square(q.x) + np.square(q.z)))

    if h < 0:
        h += 2 * np.pi

    return h, p, r


def custom_euler_to_quat(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = quaternion.quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy

    return q


def process_samples(samples: np.array, desired_length):
    logger = logging.getLogger('process_samples()')

    benchmark = time.perf_counter()

    def clock_that_step(description, benchmark):
        logger.debug('{} took {:.0f} ms'.format(description.capitalize(), (time.perf_counter() - benchmark) * 1000))
        return time.perf_counter()

    if not len(samples) > 1:
        raise AttributeError('Sample list is empty')

    logger.debug('Normalizing:\n{0!r}'.format(samples))

    # Strip redundant bearings
    unique_bearings = [samples[0]]
    for index, bearing in enumerate(samples):
        if not index:
            continue

        if np.isclose(bearing[0], samples[index - 1, 0]) and np.isclose(bearing[1], samples[index - 1, 1]):
            logger.debug('Discarding redundant point ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))
        else:
            unique_bearings.append(bearing)

    samples = np.array(unique_bearings)

    benchmark = clock_that_step('Stripping dupes', benchmark)

    # Remap standardized bearings so gestures are the same size
    yaw_min = min(samples[:, 0])
    yaw_max = max(samples[:, 0])

    pitch_min = min(samples[:, 1])
    pitch_max = max(samples[:, 1])

    magnitude = np.linalg.norm([yaw_max - yaw_min, pitch_max - pitch_min])
    fudge_factor = 1 / 10

    logger.debug('Yaw min: {:.3f} Pitch min: {:.3f} Yaw max: {:.3f} Pitch max: {:.3f} Trim length: {:.3f}'.format(
        yaw_min, pitch_min, yaw_max, pitch_max, magnitude * fudge_factor))

    early_crap_count = 0

    for i in range(1, len(samples)):
        if np.linalg.norm([samples[i, 0] - samples[0, 0],
                           samples[i, 1] - samples[0, 1]]) > magnitude * fudge_factor:
            logger.debug('Done stripping leading points - ({:.3f}, {:.3f}) is far enough from start point '
                         '({:.3f}, {:.3f}). Had to be {:.3f} units away, and is {:.3f}.'.format(
                samples[i, 0], samples[i, 1], samples[0, 0], samples[0, 1],
                magnitude * fudge_factor,
                np.linalg.norm([samples[i, 0] - samples[0, 0],
                                samples[i, 1] - samples[0, 1]])))
            break
        else:
            logger.debug('Stripping leading point ({:.3f}, {:.3f}) - too close to start point ({:.3f}, {:.3f}). '
                         'Must be {:.3f} units away, but is {:.3f}.'.format(
                samples[i, 0], samples[i, 1], samples[0, 0], samples[0, 1],
                magnitude * fudge_factor,
                np.linalg.norm([samples[i, 0] - samples[0, 0],
                                samples[i, 1] - samples[0, 1]])))
            early_crap_count += 1

    start_point = samples[0]

    trimmed = samples[early_crap_count + 1:].tolist()

    samples = np.array([start_point] + trimmed)

    benchmark = clock_that_step('Trimming early slop', benchmark)

    # logger.debug('Early crap stripped: {}'.format(samples))

    late_crap_count = 0

    for i in range(2, len(samples)):
        if np.linalg.norm([samples[-i, 0] - samples[- 1, 0],
                           samples[-i, 1] - samples[- 1, 1]]) > magnitude * fudge_factor:
            logger.debug('Done stripping trailing points - ({:.3f}, {:.3f}) is far enough from endpoint '
                         '({:.3f}, {:.3f}). Had to be {:.3f} units away, and is {:.3f}.'.format(
                samples[-i, 0], samples[-i, 1], samples[-1, 0], samples[-1, 1],
                magnitude * fudge_factor,
                np.linalg.norm([samples[-i, 0] - samples[- 1, 0],
                                samples[-i, 1] - samples[- 1, 1]])))
            break
        else:
            logger.debug('Stripping trailing point ({:.3f}, {:.3f}) - too close to endpoint ({:.3f}, {:.3f}). '
                         'Must be {:.3f} units away, but is {:.3f}.'.format(
                samples[-i, 0], samples[-i, 1], samples[-1, 0], samples[-1, 1],
                magnitude * fudge_factor,
                np.linalg.norm([samples[-i, 0] - samples[- 1, 0],
                                samples[-i, 1] - samples[- 1, 1]])))
            late_crap_count += 1

    if late_crap_count:
        endpoint = samples[-1]
        trimmed = samples[:(late_crap_count + 1) * -1].tolist()
        samples = np.array(trimmed + [endpoint])

    logger.debug('Late crap stripped: {}'.format(samples))

    benchmark = clock_that_step('Trimming late slop', benchmark)

    # Standardize bearings 'curve' to evenly-spaced points

    cumulative_segment_lengths = [0]
    for index, sample in enumerate(samples):
        if index == 0:
            continue

        segment_length = np.linalg.norm([sample[0] - samples[index - 1][0], sample[1] - samples[index - 1][1]])

        cumulative_segment_lengths.append(segment_length + cumulative_segment_lengths[index - 1])
        logger.debug('Segment ending in point {} length {:.3f} Cumul: {:.3f}'.format(
            index, segment_length, cumulative_segment_lengths[index]))

    curve_length = cumulative_segment_lengths[-1]
    target_segment_length = curve_length / (desired_length - 1)

    benchmark = clock_that_step('Calculating segment lengths', benchmark)

    # logger.debug(
    #     'Segment lengths: {} - {} segments, {} points'.format(segment_lengths, len(segment_lengths), len(samples)))
    logger.debug('Total length: {:.2f} Target segment length: {:.4f}'.format(curve_length, target_segment_length))

    standardized_bearings = [samples[0]]
    first_longer_sample = 0

    for i in range(1, desired_length):
        target_length = i * target_segment_length

        logger.debug('Looking to place a point at {:.3f} units along curve'.format(target_length))

        if cumulative_segment_lengths[first_longer_sample] > target_length:
            logger.debug('Previous point at {:.3f} units along curve still works'.format(
                cumulative_segment_lengths[first_longer_sample]))

        else:
            while cumulative_segment_lengths[first_longer_sample] < target_length \
                    and not np.isclose(cumulative_segment_lengths[first_longer_sample], target_length):
                logger.debug(
                    'Cumulative length of {:.3f} is too short - advancing to segment ending at point {}'.format(
                        cumulative_segment_lengths[first_longer_sample], first_longer_sample))
                first_longer_sample += 1

                if first_longer_sample >= len(cumulative_segment_lengths):
                    raise AttributeError("Entire line isn't long enough?!")

        low_point = samples[first_longer_sample - 1]
        high_point = samples[first_longer_sample]
        position_along_segment = ((target_length - cumulative_segment_lengths[first_longer_sample - 1]) /
                                  (cumulative_segment_lengths[first_longer_sample]
                                   - cumulative_segment_lengths[first_longer_sample - 1]))

        standardized_point_x = low_point[0] + position_along_segment * (high_point[0] - low_point[0])
        standardized_point_y = low_point[1] + position_along_segment * (high_point[1] - low_point[1])

        standardized_point = [standardized_point_x, standardized_point_y]

        logger.debug('Placed point {:.3f} units ({:.0f}%) along the {:.3f} line between {} and {} ==> {}'
                     .format(target_length - cumulative_segment_lengths[first_longer_sample - 1],
                             position_along_segment * 100,
                             cumulative_segment_lengths[first_longer_sample]
                             - cumulative_segment_lengths[first_longer_sample - 1],
                             low_point, high_point, standardized_point))

        standardized_bearings.append(standardized_point)

    logger.debug('Done interpolating. Scaling into 0-1 fractional dims')

    benchmark = clock_that_step('Interpolation', benchmark)

    # Move lowest and leftest points to the edge
    standardized_bearings = [[y - yaw_min, p - pitch_min] for y, p in standardized_bearings]

    # Rescale, preserving proportions
    total_width = yaw_max - yaw_min
    total_height = pitch_max - pitch_min

    standardized_bearings = np.array([[custom_interpolate(y, 0, max(total_width, total_height), 0, 1),
                                       custom_interpolate(p, 0, max(total_width, total_height), 0, 1)]
                                      for y, p in standardized_bearings])

    clock_that_step('Resizing', benchmark)

    return standardized_bearings


def wrapped_delta(old, new):
    delta = old - new

    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi

    return delta


def bearing_delta(old, new):
    return np.array([wrapped_delta(old[0], new[0]),
                     wrapped_delta(old[1], new[1])])


# This is taken from https://github.com/pyserial/pyserial/issues/216#issuecomment-369414522
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s  # Serial object

    def readline(self):
        timeout = self.s.timeout
        self.s.timeout = 0.1

        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i + 1]
            self.buf = self.buf[i + 1:]
            self.s.timeout = timeout
            return r

        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i + 1]
                self.buf[0:] = data[i + 1:]
                self.s.timeout = timeout
                return r
            else:
                self.buf.extend(data)
