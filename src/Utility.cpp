/*
 * @Author: clicheeeeee waterwet@outlook.com
 * @Date: 2022-06-29 16:05:35
 * @LastEditors: clicheeeeee waterwet@outlook.com
 * @LastEditTime: 2022-11-18 18:13:13
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

/**
 * split a string using the specified delimiter (can only be a single char).
 * @param input_str {const string&} 
 * @param delimiter {char} 
 * @return {vector<string>} vector of splited tokens
 */
std::vector<std::string> splitString(const std::string& input_str, char delimiter)
{
    std::stringstream ss(input_str);
    std::vector<std::string> result;

    std::string tmp_strlet;

    while (getline(ss, tmp_strlet, delimiter)) {
        result.push_back(tmp_strlet);
    }

    return result;
}


SensorType parseSensorType(std::string sensor_str)
{
    if (sensor_str.find("HDL_32E") != std::string::npos) {
        return SensorType::HDL_32E;

    } else if (sensor_str.find("HDL_64E") != std::string::npos) {
        return SensorType::HDL_64E;

    } else if (sensor_str.find("OS1_64") != std::string::npos) {
        return SensorType::OS1_64;
        
    }

    // should not reach here
    std::string err_msg = fmt::format("Unknown sensor type: {}!", sensor_str);
    std::cerr << err_msg << std::endl;
    return SensorType::UNKNOWN;
}


SensorParams getSensorParams(SensorType sensor_type)
{
    SensorParams params;

    switch (sensor_type) {
        case SensorType::HDL_32E:
            params.N_SCAN = 32;
            params.Horizon_SCAN = 1056;
            params.GROUND_UPPER_SCAN = 20;
            params.HEIGHT_RES = 0.5f;
            break;
        
        case SensorType::HDL_64E:
            params.N_SCAN = 64;
            params.Horizon_SCAN = 2083;
            params.GROUND_UPPER_SCAN = 50;
            // TODO: need to tune this param later
            params.HEIGHT_RES = 0.25f;
            break;
        
        case SensorType::OS1_64:
            params.N_SCAN = 64;
            params.Horizon_SCAN = 1024;
            params.GROUND_UPPER_SCAN = 31;
            params.HEIGHT_RES = 1.0f;
            break;

        default:
            std::cerr << "Unknown sensor type! " << std::endl;
    }

    return params;
}

std::string printSensorParams(SensorParams params)
{
    std::string msg = fmt::format("N_SCAN: {}, Horizon_SCAN: {}, GROUND_UPPER_SCAN: {}", 
            params.N_SCAN, params.Horizon_SCAN, params.GROUND_UPPER_SCAN);
    
    return msg;
}