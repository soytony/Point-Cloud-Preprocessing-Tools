/*
 * @Author: clicheeeeee waterwet@outlook.com
 * @Date: 2022-06-29 16:05:35
 * @LastEditors: clicheeeeee waterwet@outlook.com
 * @LastEditTime: 2022-07-03 20:54:51
 * @FilePath: /pointcloud_preprocessing/src/Utility.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "Utility.h"

bool isRotationMatirx(Eigen::Matrix3d R)
{
    // double err=1e-6;
    double err=1e-4;
    Eigen::Matrix3d shouldIdenity;
    shouldIdenity=R*R.transpose();
    Eigen::Matrix3d I=Eigen::Matrix3d::Identity();
    return (shouldIdenity - I).norm() < err;
}

Eigen::Vector3d rotationMatrixToEulerAngles(Eigen::Matrix3d &R)
{
    if(!isRotationMatirx(R)) {
        std::cerr << "Not A Rotation Matrix. " << std::endl;
    };
    double sy = sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular)
    {
        x = atan2( R(2,1), R(2,2));
        y = atan2(-R(2,0), sy);
        z = atan2( R(1,0), R(0,0));
    }
    else{
        x = atan2(-R(1,2), R(1,1));
        y = atan2(-R(2,0), sy);
        z = 0;
    }
    return {x, y, z};
}

float getDistance(Pose6f frame_1, Pose6f frame_2)
{
    float diff_x = frame_1.x - frame_2.x;
    float diff_y = frame_1.y - frame_2.y;
    float diff_z = frame_1.z - frame_2.z;
    return sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}
