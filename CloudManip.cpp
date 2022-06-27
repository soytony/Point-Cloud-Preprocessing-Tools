//
// Created by tony on 2022/4/14.
//

#include "CloudManip.h"

#include <iostream>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>

//#include <vtkAutoInit.h>
//VTK_MODULE_INIT(vtkRenderingOpenGL);
//VTK_MODULE_INIT(vtkInteractionStyle);
//VTK_MODULE_INIT(vtkRenderingFreeType);
#include <string>
#include <iostream>
#include <sstream>
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

using namespace std;
//typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

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

// typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZIRCT PointType;

std::vector<std::string> splitString(std::string input_str, char token)
{
    stringstream ss(input_str);
    std::vector<std::string> result;

    std::string tmp_strlet;

    while (getline(ss, tmp_strlet, token)) {
        result.push_back(tmp_strlet);
    }

    return result;
}

void saveAsMat(pcl::PointCloud<PointType>::Ptr cloud, std::string mat_filename, float interval = 2.0f)
{
    static int MAX_RANGE = 100;
    static int MAT_SIZE = MAX_RANGE*2 / interval + 1;
    cv::Mat cart_bv = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
    for (auto &pi : cloud->points) {
        int x = round((pi.x + MAX_RANGE) / interval + 0.5);
        int y = round((pi.y + MAX_RANGE) / interval + 0.5);

        if (x < 0 || x >= MAT_SIZE || y < 0 || y >= MAT_SIZE) {
            continue;
        }

        if (pi.z + 2.0f > cart_bv.at<float>(x, y)) {
            cart_bv.at<float>(x, y) = pi.z + 2.0f;
        }
    }

    cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_CSV);
    fmt->set64fPrecision(4);
    fmt->set32fPrecision(4);

    std::ofstream of(mat_filename, std::ios::out);
    if (of.is_open()) {
        of << fmt->format(cart_bv);
    } else {
        std::cerr << "Can not open file: " << mat_filename << "\n";
    }

    cv::imwrite(mat_filename+".png", cart_bv);
}

int main(int argc, char** argv)
{
    pcl::PointCloud<PointType>::Ptr cloud_input(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud_output(new pcl::PointCloud<PointType>());

    std::string input_filename(argv[1]);
    pcl::io::loadPCDFile(input_filename, *cloud_input);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    float trans_x = std::stof(argv[2]);
    float trans_y = std::stof(argv[3]);
    float trans_z = std::stof(argv[4]);
    transform.translation() << trans_x, trans_y, trans_z;
    float theta = std::stof(argv[5]) / 180.0f * M_PI;
    std::cout << "rotating yaw radiance: " << theta << "\n";
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));

    pcl::transformPointCloud(*cloud_input, *cloud_output, transform);

    std::vector<std::string> split_result = splitString(input_filename, '/');
    std::string short_name = split_result.back();

    //float interval_res = std::stof(argv[5]);
    float interval_res = 1.0f;

    saveAsMat(cloud_input, short_name + "_input.csv", interval_res);
    saveAsMat(cloud_output, short_name + "_output.csv", interval_res);

    pcl::io::savePCDFileBinary(short_name + "_input.pcd", *cloud_input);
    pcl::io::savePCDFileBinary(short_name + "_output.pcd", *cloud_output);


    pcl::visualization::PCLVisualizer viewer ("Mip Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<PointType> input_cloud_color_handler (cloud_input, 255, 0, 0);
    viewer.addPointCloud(cloud_input, input_cloud_color_handler, "cloud_input");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> output_cloud_color_handler (cloud_output, 0, 255, 0);
    viewer.addPointCloud(cloud_output, output_cloud_color_handler, "cloud_output");

    viewer.addCoordinateSystem (1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // 设置背景为深灰
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_input");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_output");
    //viewer.setPosition(800, 400); // 设置窗口位置

    while (!viewer.wasStopped ()) { // 在按下 "q" 键之前一直会显示窗口
        viewer.spinOnce ();
    }

    return 0;
}