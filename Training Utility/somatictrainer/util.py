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

    if not sum(timestamps):
        raise AttributeError('Gesture has no duration')

    scaling_factor = standard_gesture_time / timestamps[-1]
    # Standardize times by interpolating
    scaled_times = timestamps * scaling_factor

    bearing_model = interp1d(scaled_times, bearings, axis=0, kind='cubic', fill_value='extrapolate')
    accel_model = interp1d(scaled_times, raw_accel, axis=0, kind='cubic', fill_value='extrapolate')

    standardized_bearings = bearing_model(np.linspace(0, standard_gesture_time,
                                                      num=int(np.ceil(standard_gesture_time / sampling_rate)),
                                                      endpoint=True))

    standardized_accel = accel_model(np.linspace(0, standard_gesture_time,
                                                 num=int(np.ceil(standard_gesture_time / sampling_rate)),
                                                 endpoint=True))

    #
    # orientations = []
    # accelerations = []
    #
    # # Interpolate to increase/reduce number of samples to required sampling rate
    # for earliest_time in range(0, standard_gesture_time, sampling_rate):
    #     # For each sample required, find the latest sample before this time, and the earliest sample
    #     # after this time, and slerp them.
    #
    #     # TODO This can probably be done directly by numpy, figger it out
    #
    #     early_orientation = None
    #     early_acceleration = None
    #     early_time = None
    #     late_orientation = None
    #     late_acceleration = None
    #     late_time = None
    #
    #     latest_time = earliest_time + sampling_rate
    #
    #     sample_time = 0
    #     for index in range(len(timedeltas)):
    #         if index:
    #             sample_time += scaled_times[index]
    #
    #         if early_orientation is None and sample_time >= earliest_time:
    #             # This sample is the latest sample that began earlier than the early time.
    #             early_orientation = quats[index]
    #             early_acceleration = raw_accel[index]
    #             early_time = sample_time - scaled_times[index]
    #
    #         if late_orientation is None and (sample_time > latest_time or np.isclose(sample_time, latest_time)):
    #             # This sample is the latest sample that began earlier than the late time.
    #             late_orientation = quats[index]
    #             late_acceleration = raw_accel[index]
    #             late_time = sample_time
    #
    #         if early_orientation is not None and late_orientation is not None \
    #                 and early_acceleration is not None and late_acceleration is not None:
    #             break
    #
    #     if early_orientation is None or early_acceleration is None \
    #             or late_orientation is None or late_acceleration is None:
    #         raise AttributeError('Something went wrong - missing data {0} and {1}. Got {2}/{4} and {3}/{5}'
    #                              .format(earliest_time, latest_time,
    #                                      early_orientation, late_orientation,
    #                                      early_acceleration, late_acceleration))
    #
    #     # amount = (earliest_time - early_time) / (late_time - early_time)  # Just the Arduino map function
    #
    #     orientations.append(quaternion.quaternion_time_series.slerp(
    #         early_orientation, late_orientation, early_time, late_time, earliest_time))
    #
    #     # accelerations.append(np.interp(amount, [0, 1], [early_acceleration, late_acceleration]))
    #     linear_fit = interp1d([early_time, late_time], np.vstack([early_acceleration, late_acceleration]), axis=0)
    #     accelerations.append(linear_fit(earliest_time))
    #
    # bearings = np.array([custom_euler(q)[:2] for q in orientations])

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
                     wrapped_delta(old[1], new[1]),
                     wrapped_delta(old[2], new[2])])
