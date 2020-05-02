import logging
from typing import List, Tuple

import numpy as np
import quaternion
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.DEBUG)


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


def process_samples(samples: List[Tuple[np.array, np.array, int]], standard_gesture_time, sampling_rate):
    if not samples:
        raise AttributeError('Samples and/or timedeltas not provided')

    if not len(samples):
        raise AttributeError('Samples and/or timedeltas list are empty')

    logging.info('Normalizing:\n{0!r}'.format(samples))

    bearings = np.vstack([x[0] for x in samples])
    raw_accel = np.vstack([x[1] for x in samples])
    timestamps = np.array([x[2] for x in samples])

    # Strip redundant bearings
    unique_bearings = [bearings[0]]
    for index, bearing in enumerate(bearings):
        if not index:
            continue

        if np.isclose(bearing[0], bearings[index - 1, 0]) and np.isclose(bearing[1], bearings[index - 1, 1]):
            logging.debug('Discarding redundant point ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))
        else:
            unique_bearings.append(bearing)

    bearings = np.array(unique_bearings)

    # Remap standardized bearings so gestures are the same size
    yaw_min = min(bearings[:, 0])
    yaw_max = max(bearings[:, 0])

    pitch_min = min(bearings[:, 1])
    pitch_max = max(bearings[:, 1])

    magnitude = np.linalg.norm([yaw_max - yaw_min, pitch_max - pitch_min])
    fudge_factor = 1 / 10

    early_crap_count = 0

    for i in range(1, len(bearings)):
        if np.linalg.norm([bearings[i, 0] - bearings[0, 0],
                           bearings[i, 1] - bearings[0, 1]]) > magnitude * fudge_factor:
            logging.debug('Done stripping leading points - ({:.3f}, {:.3f}) is far enough from start point '
                          '({:.3f}, {:.3f}). Had to be {:.3f} units away, and is {:.3f}.'.format(
                            bearings[i, 0], bearings[i, 1], bearings[0, 0], bearings[0, 1],
                            magnitude * fudge_factor,
                            np.linalg.norm([bearings[i, 0] - bearings[0, 0],
                                            bearings[i, 1] - bearings[0, 1]])))
            break
        else:
            logging.debug('Stripping leading point ({:.3f}, {:.3f}) - too close to start point ({:.3f}, {:.3f}). '
                          'Must be {:.3f} units away, but is {:.3f}.'.format(
                            bearings[i, 0], bearings[i, 1], bearings[0, 0], bearings[0, 1],
                            magnitude * fudge_factor,
                            np.linalg.norm([bearings[i, 0] - bearings[0, 0],
                                            bearings[i, 1] - bearings[0, 1]])))
            early_crap_count += 1

    start_point = bearings[0]

    trimmed = bearings[early_crap_count:].tolist()

    bearings = np.array([start_point] + trimmed)

    logging.debug('Early crap stripped: {}'.format(bearings))

    late_crap_count = 0

    for i in range(1, len(bearings)):
        if np.linalg.norm([bearings[-i, 0] - bearings[- 1, 0],
                           bearings[-i, 1] - bearings[- 1, 1]]) > magnitude * fudge_factor:
            logging.debug('Done stripping trailing points - ({:.3f}, {:.3f}) is far enough from endpoint '
                          '({:.3f}, {:.3f}). Had to be {:.3f} units away, and is {:.3f}.'.format(
                            bearings[-i, 0], bearings[-i, 1], bearings[-1, 0], bearings[-1, 1],
                            magnitude * fudge_factor,
                            np.linalg.norm([bearings[-i, 0] - bearings[- 1, 0],
                                            bearings[-i, 1] - bearings[- 1, 1]])))
            break
        else:
            logging.debug('Stripping trailing point ({:.3f}, {:.3f}) - too close to endpoint ({:.3f}, {:.3f}). '
                          'Must be {:.3f} units away, but is {:.3f}.'.format(
                            bearings[-i, 0], bearings[-i, 1], bearings[-1, 0], bearings[-1, 1],
                            magnitude * fudge_factor,
                            np.linalg.norm([bearings[-i, 0] - bearings[- 1, 0],
                                            bearings[-i, 1] - bearings[- 1, 1]])))
            late_crap_count += 1

    if late_crap_count:
        endpoint = bearings[-1]
        trimmed = bearings[:-late_crap_count - 1].tolist()
        bearings = np.array(trimmed + [endpoint])

    logging.debug('Late crap stripped: {}'.format(bearings))

    if not sum(timestamps):
        raise AttributeError('Gesture has no duration')

    target_sample_count = int(np.ceil(standard_gesture_time / sampling_rate))

    scaling_factor = standard_gesture_time / timestamps[-1]
    # Standardize times by interpolating
    scaled_times = timestamps * scaling_factor

    accel_model = interp1d(scaled_times, raw_accel, axis=0, kind='cubic', fill_value='extrapolate')

    standardized_accel = accel_model(np.linspace(0, standard_gesture_time,
                                                 num=int(np.ceil(standard_gesture_time / sampling_rate)),
                                                 endpoint=True))

    # Standardize bearings 'curve' to evenly-spaced points
    segment_lengths = [np.linalg.norm([bearings[i, 0] - bearings[i - 1, 0], bearings[i, 1] - bearings[i - 1, 1]])
                       for i in range(1, len(bearings))]

    curve_length = sum(segment_lengths)

    target_segment_length = curve_length / (target_sample_count - 1)

    standardized_bearings = [bearings[0]]

    logging.debug(
        'Segment lengths: {} - {} segments, {} points'.format(segment_lengths, len(segment_lengths), len(bearings)))
    logging.debug('Total length: {:.2f} Target segment length: {:.4f}'.format(curve_length, target_segment_length))

    lower_length = 0
    higher_length = 0
    first_longer_sample = 0

    for i in range(1, target_sample_count):
        logging.debug('Looking to place a point at {:.3f} units along curve'.format(i * target_segment_length))

        moved_along_curve = False

        while higher_length < i * target_segment_length \
                and not np.isclose(higher_length, i * target_segment_length) \
                and not np.isclose(higher_length, curve_length):
            lower_length = higher_length
            higher_length = higher_length + segment_lengths[first_longer_sample]
            first_longer_sample += 1
            moved_along_curve = True

            logging.debug(
                'Is {:.3f} enough? If so, {} is the next longest'.format(higher_length, first_longer_sample))

        if moved_along_curve:
            logging.debug('Previous point at {:.3f} units along curve still works'.format(higher_length))
            logging.debug('There we go')

        low_point = bearings[first_longer_sample - 1]
        high_point = bearings[first_longer_sample]
        position_along_segment = (i * target_segment_length - lower_length) / (higher_length - lower_length)

        standardized_point_x = low_point[0] + position_along_segment * (high_point[0] - low_point[0])
        standardized_point_y = low_point[1] + position_along_segment * (high_point[1] - low_point[1])

        standardized_point = [standardized_point_x, standardized_point_y]

        logging.debug('Placed point {:.3f} units ({:.0f}%) along the {:.3f} line between {} and {} ==> {}'
                      .format(i * target_segment_length - lower_length,
                              position_along_segment * 100,
                              higher_length - lower_length, low_point, high_point, standardized_point))

        standardized_bearings.append(standardized_point)

    logging.debug('Done interpolating. Scaling into 0-1 fractional dims')

    # Move lowest and leftest points to the edge
    standardized_bearings = [[y - yaw_min, p - pitch_min] for y, p in standardized_bearings]

    # Rescale, preserving proportions
    total_width = yaw_max - yaw_min
    total_height = pitch_max - pitch_min

    standardized_bearings = np.array([[custom_interpolate(y, 0, max(total_width, total_height), 0, 1),
                                       custom_interpolate(p, 0, max(total_width, total_height), 0, 1)]
                                      for y, p in standardized_bearings])

    return standardized_bearings, standardized_accel


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
