#ifndef __UTILITY_HEADER__
#define __UTILITY_HEADER__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>


bool isRotationMatirx(Eigen::Matrix3d R);
Eigen::Vector3d rotationMatrixToEulerAngles(Eigen::Matrix3d &R);

#endif