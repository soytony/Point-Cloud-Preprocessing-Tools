//
// Created by tony on 22-4-19.
//

#include "KittiRawPointCloudSelect.h"

#include <iostream>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <boost/format.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>

namespace pcl {
    struct PointXYZIRCT {
        PCL_ADD_POINT4D;
        float intensity;
        uint16_t row;
        uint16_t col;
        uint32_t t;
        int16_t label;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    }EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (
        pcl::PointXYZIRCT,
        (float, x, x)
        (float, y, y)
        (float, z, z)
        (float, intensity, intensity)
        (uint16_t, row, row)
        (uint16_t, col, col)
        (uint32_t, t, t)
        (int16_t, label, label)
)

const float KEYFRAME_DIST_INTERVAL = 2.0f;

//select the dataset source here
//std::string dataset_dir_ = "/media/tony/mas_linux/Datasets/kitti/08/";
std::string dataset_dir_ = "/media/tony/mas_linux/Datasets/kitti/05/";

std::string velo_cloud_dir_ = dataset_dir_ + "velodyne/";
std::string velo_time_filename_ = dataset_dir_ + "times.txt";
std::string gt_pose_filename_ = dataset_dir_ + "global_pose.txt";

std::string output_root_dir_ = dataset_dir_ + "selected_keyframes/";
std::string output_cloud_dir_ = output_root_dir_ + "keyframe_point_cloud/";
std::string output_keyframe_pose_filename_ = output_root_dir_ + "keyframe_pose.csv";

struct Pose6f
{
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;

    Pose6f operator*(float ratio)
    {
        Pose6f new_pose;
        new_pose.x = ratio * x;
        new_pose.y = ratio * y;
        new_pose.z = ratio * z;
        new_pose.roll = ratio * roll;
        new_pose.pitch = ratio * pitch;
        new_pose.yaw = ratio * yaw;

        return new_pose;
    };

    Pose6f operator+(const Pose6f &pose)
    {
        Pose6f new_pose;
        new_pose.x = x + pose.x;
        new_pose.y = y + pose.y;
        new_pose.z = z + pose.z;
        new_pose.roll = roll + pose.roll;
        new_pose.pitch = pitch + pose.pitch;
        new_pose.yaw = yaw + pose.yaw;

        return new_pose;
    };
};

static inline float makeAngleSemiPositive(float input_angle)
{
    if (input_angle >= 360.0f) {
        return (input_angle - 360.0f);
    } else if (input_angle < 0) {
        return (input_angle + 360.0f);
    } else {
        return input_angle;
    }
}

const int N_SCAN = 64;
const int Horizon_SCAN = 2083;

std::vector<Pose6f> full_gt_poses_;
std::vector<Pose6f> selected_gt_poses_;
std::vector<int64_t> full_cloud_timestamps_;

pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloud(int64_t timestamp)
{
    boost::format fmt("%06d"); //kitti cloud has 6 digits long name
    fmt % timestamp;
    std::string current_file_name = velo_cloud_dir_ + fmt.str() + ".bin";

    //load current data
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZIRCT>);
    cloud->reserve(N_SCAN * Horizon_SCAN);

    ifstream file;
    file.open(current_file_name, ios::in|ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open point cloud file: " << current_file_name << std::endl;
        return cloud;
    }

    int k = 0;
    static const int MAX_NUM_POINTS = 64*2250;
    while(!file.eof() && k < MAX_NUM_POINTS)
    {
        pcl::PointXYZIRCT point;
        file.read(reinterpret_cast<char *>(&point.x), sizeof(float));
        file.read(reinterpret_cast<char *>(&point.y), sizeof(float));
        file.read(reinterpret_cast<char *>(&point.z), sizeof(float));
        file.read(reinterpret_cast<char *>(&point.intensity), sizeof(float));
        //point.ring = (k%64) + 1 ;
        k ++;
        cloud->push_back(point);
    }
    file.close();

    //compute azimuthal angle for each point in the cloud
    std::vector<float> azimuth_angle(cloud->points.size());
    for (int i = 0; i < cloud->points.size(); i ++) {
        azimuth_angle[i] =
                atan2(cloud->points[i].y, cloud->points[i].x) / M_PI * 180.0f;
    }

    int32_t ring_idx = -1;
    // drop some points with positive azimuth angle
    // we only start a ring from 0 degree
    if (azimuth_angle[0] > 0) {
        ring_idx = 0;
    } else {
        ring_idx = -1;
        std::cout << "The azimuthal angle of the first point in this cloud is < 0. " << "Cloud file name: " << current_file_name << "\n";
    }

    //structed points vector with fixed size of points
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr structured_cloud(new pcl::PointCloud<pcl::PointXYZIRCT>);
    structured_cloud->points.resize(N_SCAN * Horizon_SCAN, pcl::PointXYZIRCT{.intensity = -1});

    //fill row idx and col idx for each point
    float this_azimuth = 0;
    for (int i = 1; i < azimuth_angle.size(); i ++) {
        // see if new ring arrives
        if (azimuth_angle[i - 1] <= 0 && azimuth_angle[i] > 0) {
            ring_idx ++;
        }

        // compute column index for this point
        this_azimuth = makeAngleSemiPositive(azimuth_angle[i]);
        int col_idx = static_cast<int>(std::round(this_azimuth / (360.0 / Horizon_SCAN)));

        if (ring_idx >= 0 && ring_idx < N_SCAN) {
            if (col_idx >= Horizon_SCAN) {
                col_idx = col_idx - Horizon_SCAN;
            } else if (col_idx < 0) {
                col_idx = col_idx + Horizon_SCAN;
            }

            cloud->points[i].row = static_cast<uint16_t>(ring_idx);
            cloud->points[i].col = static_cast<uint16_t>(col_idx);
            cloud->points[i].label = static_cast<int16_t>(-2); //-2 means not segmented points

            structured_cloud->points[ring_idx * Horizon_SCAN + col_idx] = cloud->points[i];
        }
    }

    return structured_cloud;
}

void readFullGtPoses()
{
    std::fstream f_gt_pose;
    f_gt_pose.open(gt_pose_filename_, ios::in);

    if (f_gt_pose.is_open()) {
        std::cout << "loaded gt pose file: " << gt_pose_filename_ << std::endl;
    } else {
        std::cerr << "failed to load gt pose file: " << gt_pose_filename_ << std::endl;
        exit(1);
    }


    std::string tmp_pose_str;
    while (getline(f_gt_pose, tmp_pose_str)) {
        Eigen::Matrix<double,4,4> T = Eigen::Matrix<double,4,4>::Zero();
        T(3,3) = 1.0;

        std::vector<std::string> row;

        std::stringstream  ss(tmp_pose_str);
        std::string str;

        int32_t read_count = 12;
        while (read_count > 0) {
            ss >> str;
            row.push_back(str);
            read_count --;
        }
        //kitti gt has 12 fields per row
        if(row.size()!=12)
            break;

        for(int i=0;i<3;i++){
            for(int j=0;j<4;j++){
                double d = std::stod(row[(4*i)+j]);
                T(i,j) = d;
            }
        }

        //Eigen::Affine3d this_pose(T);
        Eigen::Matrix3d rotation_matrix = T.block<3,3>(0,0);
        Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0); // ZYX order: yaw pitch roll
        Pose6f this_pose6f{
                .x = float(T(0,3)),
                .y = float(T(2,3)), // note that y and x are reversed
                .z = float(T(1,3)),
                .roll = float(euler_angles(2)),
                .pitch = float(euler_angles(1)),
                .yaw = float(euler_angles(0))
        };

        full_gt_poses_.emplace_back(std::move(this_pose6f));
    }

    f_gt_pose.close();

    std::cout << "Finish reading all gt pose, total " << full_gt_poses_.size() << " entries. " << std::endl;
}

void readFullCloudTimestamps()
{
    std::fstream f_cloud_time;
    f_cloud_time.open(velo_time_filename_, ios::in);


    if(f_cloud_time.is_open()) {
        std::cout << "loaded cloud timestamps: " << velo_time_filename_ << std::endl;
    } else {
        std::cerr << "failed to load cloud timestamps: " << velo_time_filename_ << std::endl;
        exit(1);
    }


    std::string tmp_cloud_timestamp_str;
    while (f_cloud_time >> tmp_cloud_timestamp_str) {

        int64_t tmp_cloud_timestamp_i64 = std::stoll(tmp_cloud_timestamp_str);
        full_cloud_timestamps_.emplace_back(tmp_cloud_timestamp_i64);

    }

    f_cloud_time.close();

    //std::sort(full_cloud_timestamps_.begin(), full_cloud_timestamps_.end(), [](auto &time_1, auto &time_2) -> bool {
    //    return time_1 < time_2;
    //});

    std::cout << "Finish reading all cloud timestamps, total " << full_cloud_timestamps_.size() << " entries. " << std::endl;
}

float getDistBtwPoses(const Pose6f &pose_1, const Pose6f &pose_2)
{
    float diff_x = pose_2.x - pose_1.x;
    float diff_y = pose_2.y - pose_1.y;
    float diff_z = pose_2.z - pose_1.z;
    return std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}

std::string padString(int keyframe_idx)
{
    boost::format fmt("%06d");
    fmt % keyframe_idx;
    return fmt.str();
}

int main(int argc, char** argv)
{
    int unused;
    unused = system((std::string("exec rm -r ") + output_root_dir_).c_str());
    unused = system((std::string("mkdir -p ") + output_root_dir_).c_str());

    unused = system((std::string("exec rm -r ") + output_cloud_dir_).c_str());
    unused = system((std::string("mkdir -p ") + output_cloud_dir_).c_str());

    readFullGtPoses();
    readFullCloudTimestamps();

    if (full_gt_poses_.size() != full_cloud_timestamps_.size()) {
        std::cerr << "Numbers of gt poses do NOT agree with the number of velodyne point clouds. \n";
        exit(1);
    }

    std::ofstream f_keyframe_poses(output_keyframe_pose_filename_, ios::out);
    if (!f_keyframe_poses.is_open()) {
        std::cerr << "Failed to create keyframe pose file: " << output_keyframe_pose_filename_ << std::endl;
        exit(1);
    }

    int keyframe_idx = 0;

    Pose6f last_cloud_pose{.x = -1e10, .y = -1e10, .z = 0, .roll = 0, .pitch = 0, .yaw = 0};
    size_t last_gt_pose_idx = 1;
    for (size_t cloud_idx = 0; cloud_idx < full_cloud_timestamps_.size(); cloud_idx ++) {

        Pose6f cloud_pose = full_gt_poses_[cloud_idx];

        float dist_to_last_keyframe = getDistBtwPoses(cloud_pose, last_cloud_pose);
        if (dist_to_last_keyframe < KEYFRAME_DIST_INTERVAL) {
            //NOT a keyframe
            continue;
        }

        // save keyframe cloud and its pose to local
        std::cout << "Saving keyframe: " << keyframe_idx << ", dist to last keyframe: " << dist_to_last_keyframe << std::endl;
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr this_cloud = extractPointCloud(cloud_idx);
        pcl::io::savePCDFileBinary(output_cloud_dir_ + padString(keyframe_idx) + ".pcd", *this_cloud);
        boost::format fmt("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n");
        fmt % cloud_pose.x % cloud_pose.y % cloud_pose.z % cloud_pose.roll % cloud_pose.pitch % cloud_pose.yaw;
        f_keyframe_poses << fmt.str();
        selected_gt_poses_.push_back(cloud_pose);

        keyframe_idx ++;
        last_cloud_pose = cloud_pose;

        this_cloud.reset();

    }

    f_keyframe_poses.close();

    std::cout << "Done. " << std::endl;

    return 0;
}