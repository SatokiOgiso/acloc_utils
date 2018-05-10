#!/usr/bin/env python

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray


def euler_to_quaternion(euler):
    """Convert Euler Angles to Quaternion

    :param euler: geometry_msgs/Vector3
    :return quaternion: geometry_msgs/Quaternion
    """
    q = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def pack_location_PoseWithCovarianceStamped(localizer,
                                            stamp=None, frame_id="/world"):
    """ compose PoseWithCovarianceStamped from ekf_localizar(in private repo)

    :param localizer: instance of RobotEKFLocalizer
    :param stamp: ROS timestamp to set in packed message header
    :param frame_id: frame_id to set in packed message header
    :return location: PoseWithCovarianceStamped message with given data
    """

    location = PoseWithCovarianceStamped()
    if stamp is None:
        stamp = rospy.Time.now()
    location.header.stamp = stamp
    location.header.frame_id = frame_id
    location.pose.pose.position.x = localizer.x[0].copy()
    location.pose.pose.position.y = localizer.x[1].copy()
    location.pose.pose.orientation = \
        euler_to_quaternion([0, 0, localizer.x[2].copy()])
    if hasattr(localizer, "P"):
        location.pose.covariance = \
            convertCovariance2Dto3D(localizer.P).flatten("F")

    return location


def pack_location_PoseArray(particles, stamp=None, frame_id="/world"):
    """ compose PoseArray from pf_localizar(in private repo)

    :param particles: instance of RobotPFLocalizer.Particles
    :param stamp: ROS timestamp to set in packed message header
    :param frame_id: frame_id to set in packed message header
    :return location: PoseArray message with given data
    """

    location_array = PoseArray()
    if stamp is None:
        stamp = rospy.Time.now()
    location_array.header.stamp = stamp
    location_array.header.frame_id = frame_id
    for particle in particles:
        p = Pose()
        p.position.x = particle.pos[0].copy()
        p.position.y = particle.pos[1].copy()
        p.orientation = euler_to_quaternion([0, 0, particle.pos[2].copy()])
        location_array.poses.append(p)

    return location_array


def convertCovariance2Dto3D(covariance2d):
    """ convert the covariance from [x, y, theta] to [x, y, z, roll, pitch, yaw]

    :param covariance2d: covariance matrix in 3x3 format.
                         each row and column corresponds to [x, y, theta]
    :return: covariance matrix in 6x6 format. each row and column corresponds to
             [x, y, z, roll, pitch, yaw],
             where z, roll and pitch values are padded with 0.
    """

    covariance3d = np.zeros([6, 6])
    covariance2d = np.array(covariance2d)
    covariance3d[0:1, 0:1] = covariance2d[0:1, 0:1]
    covariance3d[5, 0:1] = covariance2d[2, 0:1]
    covariance3d[0:1, 5] = covariance2d[0:1, 2]
    covariance3d[5, 5] = covariance2d[2, 2]

    return covariance3d


def scale_data(data, maxval=100):
    """ scale the data into 0 to maxval

    :param data: data to be scaled.
    :param maxval: maximum value of the scaling range
    :return: scaled data
    """

    max_data = np.max(data)
    min_data = np.min(data)
    scale = max_data - min_data

    image_sc = maxval * (data - min_data) / scale

    return image_sc.astype("uint8")
