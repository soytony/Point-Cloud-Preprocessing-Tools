#ifndef __UTILITY_HEADER__
#define __UTILITY_HEADER__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <fmt/core.h>
#include <fmt/format.h>

bool isRotationMatirx(Eigen::Matrix3d R);
Eigen::Vector3d rotationMatrixToEulerAngles(Eigen::Matrix3d &R);

enum SensorType
{
    HDL_32E = 0,
    HDL_64E,
    OS1_64,
    UNKNOWN
};

struct SensorParams
{
    int Horizon_SCAN;
    int N_SCAN;
    int GROUND_UPPER_SCAN;
};

struct Pose6f
{
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;
    Eigen::Matrix3d rotation_matrix;
    Eigen::Quaterniond rotation_quat;

    // interpolate pose using linear for position
    // and slerp for orientation
    Pose6f interpolate(const Pose6f &pose_2, double ratio)
    {
        Pose6f new_pose;

        new_pose.x = x * (1-ratio) + pose_2.x * ratio;
        new_pose.y = y * (1-ratio) + pose_2.y * ratio;
        new_pose.z = z * (1-ratio) + pose_2.z * ratio;

        new_pose.rotation_quat = rotation_quat.slerp(ratio, pose_2.rotation_quat);
        new_pose.rotation_matrix = new_pose.rotation_quat.toRotationMatrix();

        // do NOT use the built-in conversion for euler angles in Eigen
        // Eigen::Vector3d euler_angles = new_pose.rotation_matrix.eulerAngles(2,1,0);
        Eigen::Vector3d euler_angles = rotationMatrixToEulerAngles(new_pose.rotation_matrix);

        new_pose.yaw = euler_angles(2);
        new_pose.pitch = euler_angles(1);
        new_pose.roll = euler_angles(0);

        return new_pose;
    };

    std::vector<float> getPositionVec()
    {
        return std::vector<float>{x,y,z};
    };
};


float getDistance(Pose6f frame_1, Pose6f frame_2);
std::vector<std::string> splitString(std::string input_str, char delimiter);
SensorType parseSensorType(std::string sensor_str);
SensorParams getSensorParams(SensorType sensor_type);
std::string printSensorParams(SensorParams params);

#endif