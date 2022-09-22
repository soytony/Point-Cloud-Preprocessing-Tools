//
// Created by tony on 22-4-23.
//

#include "OxfordPointCloudSelect.h"

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
#include <fmt/core.h>
#include <fmt/format.h>

namespace pcl {
    struct PointXYZIRCT {
        PCL_ADD_POINT4D;
        float intensity;
        std::uint16_t row;
        std::uint16_t col;
        std::uint32_t t;
        std::int16_t label;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    }EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (
    pcl::PointXYZIRCT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint16_t, row, row)
    (std::uint16_t, col, col)
    (std::uint32_t, t, t)
    (std::int16_t, label, label)
)

float KEYFRAME_DIST_INTERVAL = 2.0f;

//select the dataset source here
std::string dataset_dir_ = "/media/tony/mas_linux/Datasets/oxford/2019-01-11-13-24-51-radar-oxford-10k/";

std::string velodyne_cloud_dir_ = dataset_dir_ + "velodyne_left/";
std::string ouster_time_filename_ = dataset_dir_ + "velodyne_left.timestamps";
std::string gt_pose_filename_ = dataset_dir_ + "/gps/ins.csv";

std::string output_root_dir_ = dataset_dir_ + "selected_keyframes/";
std::string output_cloud_dir_ = output_root_dir_ + "keyframe_point_cloud/";
std::string output_keyframe_pose_data_filename_ = output_root_dir_ + "keyframe_pose.csv";
std::string output_keyframe_pose_format_filename_;

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
        Eigen::Vector3d euler_angles = new_pose.rotation_matrix.eulerAngles(2,1,0);
        new_pose.yaw = euler_angles(0);
        new_pose.pitch = euler_angles(1);
        new_pose.roll = euler_angles(2);

        return new_pose;
    };
};

void initDirectories(std::string root_dir)
{
    if (root_dir.back() != '/') {
        root_dir = root_dir + "/";
    }
    dataset_dir_ = root_dir;

    velodyne_cloud_dir_ = dataset_dir_ + "velodyne_left/";
    ouster_time_filename_ = dataset_dir_ + "velodyne_left.timestamps";
    gt_pose_filename_ = dataset_dir_ + "/gps/ins.csv";

    output_root_dir_ = fmt::format("{}selected_keyframes_{:2.2f}m/", dataset_dir_, KEYFRAME_DIST_INTERVAL);
    output_cloud_dir_ = output_root_dir_ + "keyframe_point_cloud/";
    output_keyframe_pose_data_filename_ = output_root_dir_ + "keyframe_pose.csv";
    output_keyframe_pose_format_filename_ = output_root_dir_ + "keyframe_pose_format.csv";


    int unused __attribute__((unused));
    unused = system((std::string("exec rm -r ") + output_root_dir_).c_str());
    unused = system((std::string("mkdir -p ") + output_root_dir_).c_str());

    unused = system((std::string("exec rm -r ") + output_cloud_dir_).c_str());
    unused = system((std::string("mkdir -p ") + output_cloud_dir_).c_str());
}

std::vector<std::pair<int64_t, Pose6f>> full_gt_poses_;
std::vector<Pose6f> selected_gt_poses_;
std::vector<int64_t> full_cloud_timestamps_;

std::vector<std::string> splitString(std::string input_str, char delim)
{
    std::stringstream ss(input_str);
    std::vector<std::string> result;

    std::string tmp_strlet;

    while (getline(ss, tmp_strlet, delim)) {
        result.push_back(tmp_strlet);
    }

    return result;
}

pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloud(int64_t timestamp)
{
    boost::format fmt("%010ld");
    fmt % timestamp;
    std::string current_file_name = velodyne_cloud_dir_ + fmt.str() + ".bin";

    //load current data
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZIRCT>);

    ifstream file;
    file.open(current_file_name, ios::in|ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open point cloud file: " << current_file_name << std::endl;
        return cloud;
    }

    //count the num of points in this cloud
    size_t num_points = 0;
    long start_pos, end_pos;
    ifstream f_file_size(current_file_name, ios::in | ios::binary);
    start_pos = f_file_size.tellg();
    f_file_size.seekg(0, ios::end);
    end_pos = f_file_size.tellg();
    f_file_size.close();
    num_points = (end_pos - start_pos) / (4 * sizeof(float));

    // points data are organized in the N*4 order
    // that is, say, we have N points in this cloud, the data for each point i is stored in the fashion of
    // x_0, x_1, ..., x_(N-1), y_0, y_1, ..., y_(N-1), z_0, z_1, ..., z_(N-1), intensity_0, intensity_1, ..., intensity_(N-1)

    for (size_t i = 0; i < num_points; ++i) {
        if (file.good() && !file.eof())
        {
            pcl::PointXYZIRCT point;
            file.read(reinterpret_cast<char *>(&point.x), sizeof(float));
            cloud->push_back(point);
        }
    }
    for (size_t i = 0; i < num_points; ++i) {
        if (file.good() && !file.eof()) {
            file.read(reinterpret_cast<char *>(&cloud->points[i].y), sizeof(float));
        }
    }
    for (size_t i = 0; i < num_points; ++i) {
        if (file.good() && !file.eof()) {
            file.read(reinterpret_cast<char *>(&cloud->points[i].z), sizeof(float));
        }
    }
    for (size_t i = 0; i < num_points; ++i) {
        if (file.good() && !file.eof()) {
            file.read(reinterpret_cast<char *>(&cloud->points[i].intensity), sizeof(float));
        }
    }

    //compute row and col for each point
    for (auto &point : cloud->points) {
        //the lidar is mounted upside-down
        point.x = -point.x;
        point.z = -point.z;

        point.label = -2;

        float elevation_angle = atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) / M_PI * 180.0f;
        int row_idx = round((-elevation_angle + 10.67) / 1.3335); //top-down [0,31]
        row_idx = std::min(31, std::max(0, row_idx));
        point.row = static_cast<uint16_t>(row_idx);

        float azimuthal_angle = atan2(point.y, point.x) / M_PI * 180.0f;
        if (azimuthal_angle > 360.0f) {azimuthal_angle = azimuthal_angle - 360.0f;}
        else if (azimuthal_angle < 0.0f) {azimuthal_angle = azimuthal_angle + 360.0f;}
        point.col = static_cast<uint16_t>(round(azimuthal_angle / 360.0f * 1056));
        if (point.col >= 1056) {point.col -= 1056;}
        if (point.col < 0) {point.col += 1056;}
    }

    file.close();

    return cloud;
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

    // note: oxford fused imu data format
    // timestamp,          ins_status,       latitude, longitude,altitude,  northing,      easting,      down,       utm_zone,velocity_north,velocity_east,velocity_down,roll,pitch,yaw
    // 0001547213094015673,INS_SOLUTION_GOOD,51.760642,-1.260381,114.696263,5735848.358489,620057.410348,-114.696263,30U,0.352838,-0.028782,0.011870,0.034337,-0.018738,0.047108

    std::string tmp_pose_str;
    f_gt_pose >> tmp_pose_str; // first line is usesless

    while (f_gt_pose >> tmp_pose_str) {
        std::vector<std::string> this_pose_tokens = splitString(tmp_pose_str, ',');

        int64_t stamp_int = std::stoll(this_pose_tokens[0]);
        float roll = std::stof(this_pose_tokens[14]);
        float pitch = std::stof(this_pose_tokens[13]);
        float yaw = std::stof(this_pose_tokens[12]);

        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
        Eigen::Quaterniond rotation_quat(rotation_matrix);
        Pose6f this_pose6f{
                .x = std::stof(this_pose_tokens[6]), // easting
                .y = std::stof(this_pose_tokens[5]), // northing
                .z = std::stof(this_pose_tokens[4]), // altitude
                .roll = roll,
                .pitch = pitch,
                .yaw = yaw,
                .rotation_matrix = rotation_matrix,
                .rotation_quat = rotation_quat
        };

        full_gt_poses_.emplace_back(stamp_int, this_pose6f);
    }

    f_gt_pose.close();

    std::sort(full_gt_poses_.begin(), full_gt_poses_.end(), [](auto &pair_1, auto &pair_2) -> bool {
        return pair_1.first < pair_2.first;
    });

    std::cout << "Finish reading all gt pose, total " << full_gt_poses_.size() << " entries. " << std::endl;
}

void readFullCloudTimestamps()
{
    std::fstream f_cloud_time;
    f_cloud_time.open(ouster_time_filename_, ios::in);


    if(f_cloud_time.is_open()) {
        std::cout << "loaded cloud timestamps: " << ouster_time_filename_ << std::endl;
    } else {
        std::cerr << "failed to load cloud timestamps: " << ouster_time_filename_ << std::endl;
        exit(1);
    }

    //note: oxford velodyne timestamps has the format for each entry:
    //1547213095829598 1
    // that is, timestamp for this lidar scan, and an unused number

    std::string tmp_cloud_timestamp_str;
    int unused_flag;
    while (f_cloud_time >> tmp_cloud_timestamp_str >> unused_flag) {

        int64_t tmp_cloud_timestamp_i64 = std::stoll(tmp_cloud_timestamp_str);
        full_cloud_timestamps_.emplace_back(tmp_cloud_timestamp_i64);

    }

    f_cloud_time.close();

    std::sort(full_cloud_timestamps_.begin(), full_cloud_timestamps_.end(), [](auto &time_1, auto &time_2) -> bool {
        return time_1 < time_2;
    });

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
    // Step 1: Read input arguments and do global var initialization
    if (argv[1] == nullptr) {
    std::string usage_prompt = fmt::format(
"\
Usage: {} <dataset_root_dir> keyframe_dist_interval(default=2)\n\n \
<dataset_root_dir> should be organized as follows: \n \
<dataset_root_dir>\n \
├ velodyne_left/ \n \
├ velodyne_left.timestamps \n \
└ gps/ \n \
  └ ins.csv \n \
\n\n \
This binary selects keyframes based on the interval distance between two consecutive frames. \
You may specify the interval mannualy, e.g. 1 or 2, in meters. \
After running the binary, you will have files organized as follows: \n \
<keyframes_root_dir>\n \
├ ... \n \
└ selected_keyframes_xxm/ \n \
  ├ keyframe_point_cloud/ <- folder for keyframe point clouds in pcd format \n \
  ├ keyframe_pose.csv <- gt poses for all selected keyframes \n \
  └ keyframe_pose_format.csv <- format for one entry in the gt poses \n \
", argv[0]);
        std::cout << usage_prompt << std::endl;
        exit(1);
    }
    
    if (argv[2] != nullptr) {
        KEYFRAME_DIST_INTERVAL = std::stof(std::string(argv[2]));
    }
    std::cout << "Using keyframe_dist_interval = " << KEYFRAME_DIST_INTERVAL << "m. \n";
    initDirectories(std::string(argv[1]));
    std::cout << "Using dataset_dir = " << dataset_dir_ << " \n";


    // Step 2: Read KITTI GT poses and timestamps
    readFullGtPoses();
    readFullCloudTimestamps();

    std::ofstream f_keyframe_poses(output_keyframe_pose_data_filename_, ios::out);
    if (!f_keyframe_poses.is_open()) {
        std::cerr << "Failed to create keyframe pose file: " << output_keyframe_pose_data_filename_ << std::endl;
        exit(1);
    }


    // Step 3: Save output pose format descriptions
    std::ofstream f_keyframe_poses_format(output_keyframe_pose_format_filename_, ios::out);
    if (!f_keyframe_poses_format.is_open()) {
        std::cerr << "Failed to create keyframe pose format file: " << output_keyframe_pose_format_filename_ << std::endl;
        exit(1);
    }
    f_keyframe_poses_format << 
            "cloud_idx, x, y, z, roll, pitch, yaw, \
             rotation_matrix(0 0), rotation_matrix(0 1), rotation_matrix(0 2), \
             rotation_matrix(1 0), rotation_matrix(1 1), rotation_matrix(1 2), \
             rotation_matrix(2 0), rotation_matrix(2 1), rotation_matrix(2 2)" 
            << std::endl;
    f_keyframe_poses_format.close();


    // Step 4: Select keyframes and save their interpolated gt poses
    int keyframe_idx = 0;

    Pose6f last_cloud_pose{.x = 0, .y = 0, .z = 0, .roll = 0, .pitch = 0, .yaw = 0};
    size_t last_gt_pose_idx = 1;
    for (size_t cloud_idx = 0; cloud_idx < full_cloud_timestamps_.size(); cloud_idx ++) {
        int64_t this_cloud_time = full_cloud_timestamps_[cloud_idx];
        int64_t this_pose_time = -1;
        int64_t last_pose_time = -1;
        Pose6f begin_pose, end_pose; // for interpolation
        bool if_found_pose = false;
        for (size_t gt_pose_idx = last_gt_pose_idx; gt_pose_idx < full_gt_poses_.size(); gt_pose_idx ++) {
            this_pose_time = full_gt_poses_[gt_pose_idx].first;
            last_pose_time = full_gt_poses_[gt_pose_idx - 1].first;
            if (last_pose_time <= this_cloud_time && this_pose_time >= this_cloud_time) {
                last_gt_pose_idx = gt_pose_idx;
                if_found_pose = true;
                begin_pose = full_gt_poses_[gt_pose_idx - 1].second;
                end_pose = full_gt_poses_[gt_pose_idx].second;
                break;
            }
        }

        if (this_pose_time == -1 || last_pose_time == -1 || !if_found_pose) {
            std::cerr << "Could not find pose for cloud at timestamp: " << this_cloud_time << std::endl;
            continue;
        }

        double lamda = double(this_cloud_time - last_pose_time) / double(this_pose_time - last_pose_time);
        //Pose6f cloud_pose = begin_pose * (1 - lamda) + end_pose * lamda;
        //when lamda equals 1.0, cloud_pose is equal to the end_pose
        Pose6f cloud_pose = begin_pose.interpolate(end_pose, lamda);

        float dist_to_last_keyframe = getDistBtwPoses(cloud_pose, last_cloud_pose);
        if (dist_to_last_keyframe < KEYFRAME_DIST_INTERVAL) {
            //NOT a keyframe
            continue;
        }

        // save keyframe cloud and its pose to local
        std::cout << "Saving keyframe: " << keyframe_idx << ", dist to last keyframe: " << dist_to_last_keyframe << std::endl;
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr this_cloud = extractPointCloud(this_cloud_time);
        pcl::io::savePCDFileBinary(output_cloud_dir_ + padString(keyframe_idx) + ".pcd", *this_cloud);
        std::string pose_entry = 
        fmt::format("{:06d},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n",
            cloud_idx, cloud_pose.x, cloud_pose.y, cloud_pose.z, cloud_pose.roll, cloud_pose.pitch, cloud_pose.yaw, 
            cloud_pose.rotation_matrix(0,0), cloud_pose.rotation_matrix(0,1), cloud_pose.rotation_matrix(0,2), 
            cloud_pose.rotation_matrix(1,0), cloud_pose.rotation_matrix(1,1), cloud_pose.rotation_matrix(1,2), 
            cloud_pose.rotation_matrix(2,0), cloud_pose.rotation_matrix(2,1), cloud_pose.rotation_matrix(2,2));
        f_keyframe_poses << pose_entry;
        selected_gt_poses_.push_back(cloud_pose);

        keyframe_idx ++;
        last_cloud_pose = cloud_pose;

    }

    f_keyframe_poses.close();

    std::cout << "Done. " << std::endl;

    return 0;
}