/*
 * @Author: clicheeeeee waterwet@outlook.com
 * @Date: 2022-04-17 16:38:04
 * @LastEditors: clicheeeeee waterwet@outlook.com
 * @LastEditTime: 2022-09-22 20:00:10
 * @FilePath: /pointcloud_preprocessing/BatchCloudManip.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
//
// Created by tony on 22-4-17.
//

#ifndef POINTCLOUD_PCA_TEST_BATCHCLOUDMANIP_H
#define POINTCLOUD_PCA_TEST_BATCHCLOUDMANIP_H

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>

#include <dirent.h>
#include <stdlib.h>

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

//using namespace std;
//typedef pcl::PointXYZ PointType;
//typedef pcl::Normal NormalType;


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


typedef pcl::PointXYZIRCT PointType;



inline std::pair<int, int> getBelongingGrid(
        const pcl::PointCloud<PointType>::Ptr & cloud_ptr, int point_index) {
    int sector_row_idx = 0;
    int sector_col_idx = 0;

    float normalized_x = cloud_ptr->points[point_index].x + 75.0;
    float normalized_y = cloud_ptr->points[point_index].y + 50.0;

    sector_row_idx = static_cast<int>(std::floor(normalized_x / 2.0));
    sector_col_idx = static_cast<int>(std::floor(normalized_y / 2.0));

    if (sector_row_idx >= 75) {
        sector_row_idx = 75 - 1;
    }
    if (sector_row_idx < 0) {
        sector_row_idx = 0;
    }

    if (sector_col_idx >= 50) {
        sector_col_idx = 50 - 1;
    }
    if (sector_col_idx < 0) {
        sector_col_idx = 0;
    }

    return std::make_pair(sector_row_idx, sector_col_idx);
}


class BatchCloudManip {

};

#endif //POINTCLOUD_PCA_TEST_BATCHCLOUDMANIP_H
