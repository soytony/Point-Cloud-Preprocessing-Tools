
#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>

#include <dirent.h>
#include <stdlib.h>

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include "./include/Normal2dEstimation.h"

#include <Eigen/Core>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>


#define BACKWARD_HAS_DW 1

//using namespace std;
//typedef pcl::PointXYZ PointType;
//typedef pcl::Normal NormalType;


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


typedef pcl::PointXYZIRCT PointType;


struct IcpAlignResult 
{
    bool is_converged;
    double fitness_score;
    Eigen::Matrix4f final_transformation;
};

void extractTopAndFlatten(
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &cloud_input,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_output)
{
    static int NUM_GRID_X = 10;
    static int NUM_GRID_Y = 10;
    static float MAX_RADIUS_X = 100.0f;
    static float MAX_RADIUS_Y = 100.0f;
    float GRID_RES_X = 2.0f * MAX_RADIUS_X / NUM_GRID_X;
    float GRID_RES_Y = 2.0f * MAX_RADIUS_Y / NUM_GRID_Y;

    static int MIN_GRID_POINTS_SIZE = 20;

    std::vector<std::vector<std::vector<int>>> grid_map_indices(NUM_GRID_X, std::vector<std::vector<int>>(NUM_GRID_Y));

    for (int point_idx = 0; point_idx < cloud_input->points.size(); point_idx ++) {
        pcl::PointXYZIRCT &point = cloud_input->points[point_idx];

        //skip ground points
        if (point.label == 0) {
            continue;
        }

        int grid_x = round((point.x + MAX_RADIUS_X) / GRID_RES_X);
        int grid_y = round((point.y + MAX_RADIUS_Y) / GRID_RES_Y);

        if (grid_x < 0 || grid_x >= NUM_GRID_X || grid_y < 0 || grid_y >= NUM_GRID_Y) {
            continue;
        }

        grid_map_indices[grid_x][grid_y].push_back(point_idx);
    }

    // sort points in each grid
    for (int grid_x = 0; grid_x < NUM_GRID_X; grid_x ++) {
        for (int grid_y = 0; grid_y < NUM_GRID_Y; grid_y ++) {
            auto &this_grid = grid_map_indices[grid_x][grid_y];

            int num_points_in_grid = this_grid.size();
            int num_points_needed = round(0.2f * num_points_in_grid); // only top 20% points
            if (num_points_in_grid < MIN_GRID_POINTS_SIZE) {
                continue;
            }

            //sort descendingly according to the altitude of each point
            std::sort(this_grid.begin(), this_grid.end(), [&cloud_input](int idx_1, int idx_2) -> bool {
                return cloud_input->points[idx_1].z > cloud_input->points[idx_2].z;
            });

            for (int i = 0; i < num_points_needed; i ++) {
                pcl::PointXYZ flat_point;
                flat_point.getArray3fMap() = cloud_input->points[this_grid[i]].getArray3fMap();
                flat_point.z = 0;
                cloud_output->push_back(flat_point);
            }
        }
    }
}


typedef pcl::PointXYZIRCT PointT;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;


void addNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        pcl::PointCloud<pcl::Normal>::Ptr normals,
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals
)
{
    normals.reset( new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr searchTree (new pcl::search::KdTree<pcl::PointXYZ>);

    Normal2dEstimation norm_est; //该对象用于计算法向量
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
    norm_est.setInputCloud(cloud);
    norm_est.setSearchMethod(searchTree);
    norm_est.setRadiusSearch(2.0);   
    // norm_est.setKSearch(4); 
    norm_est.compute(normals); //计算法向量，并存储在points_with_normals_src

    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloud, *normals, *tmp_cloud_with_normals);

    cloud_with_normals->reserve(tmp_cloud_with_normals->size());

    for (int i = 0; i < tmp_cloud_with_normals->size(); i ++) {
        auto point = tmp_cloud_with_normals->points[i];

        // if (std::isnan(point.normal_x) || std::isnan(point.normal_y) || std::isnan(point.normal_z) ||
        //     std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
        //     continue;
        // }

        // if (std::isinf(point.normal_x) || std::isinf(point.normal_y) || std::isinf(point.normal_z) ||
        //     std::isinf(point.x) || std::isinf(point.y) || std::isinf(point.z)) {
        //     continue;
        // }

        cloud_with_normals->push_back(point);
    }
}


IcpAlignResult performCoarseIcp(
    pcl::PointCloud<pcl::PointNormal>::Ptr &points_with_normals_src,
    pcl::PointCloud<pcl::PointNormal>::Ptr &points_with_normals_tgt,
    pcl::PointCloud<pcl::PointNormal>::Ptr &points_with_normals_src_aligned,
    Eigen::Matrix4f initial_guess
    )
{
    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;

    icp.setMaxCorrespondenceDistance(10.0f);
    // typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane; 
    // boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
    // icp.setTransformationEstimation(point_to_plane);
    icp.setMaximumIterations(10);
    // icp.setTransformationEpsilon(1e-2);
    // icp.setEuclideanFitnessEpsilon(1e-2);
    // icp.setRANSACIterations(0);

    icp.setInputSource(points_with_normals_src);
    icp.setInputTarget(points_with_normals_tgt);
    
    icp.align(*points_with_normals_src_aligned, initial_guess);

    IcpAlignResult result{
        .is_converged = icp.hasConverged(),
        .fitness_score = icp.getFitnessScore(),
        .final_transformation = icp.getFinalTransformation()};

    return result;
}


IcpAlignResult performFineIcp(
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &full_cloud_1_ds,
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &full_cloud_2_ds,
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &full_cloud_1_ds_aligned,
    Eigen::Matrix4f initial_guess
)
{
    pcl::IterativeClosestPoint<pcl::PointXYZIRCT, pcl::PointXYZIRCT> icp_full;
    icp_full.setMaxCorrespondenceDistance(1.0f);
    icp_full.setTransformationEpsilon(1e-6);
    icp_full.setEuclideanFitnessEpsilon(0.01);
    icp_full.setMaximumIterations (100);

    icp_full.setInputSource(full_cloud_1_ds);
    icp_full.setInputTarget(full_cloud_2_ds);
    icp_full.align(*full_cloud_1_ds_aligned, initial_guess);

    IcpAlignResult result{
        .is_converged = icp_full.hasConverged(),
        .fitness_score = icp_full.getFitnessScore(),
        .final_transformation = icp_full.getFinalTransformation()};

    return result;
}

void separatePointsAndNormals(
    pcl::PointCloud<pcl::PointNormal>::Ptr pointnormals,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points,
    pcl::PointCloud<pcl::Normal>::Ptr normals
)
{
    points->resize(pointnormals->size());
    normals->resize(pointnormals->size());

    for (int i = 0; i < pointnormals->size(); i++) {
        points->points[i].getArray3fMap() = pointnormals->points[i].getArray3fMap();
        normals->points[i].getNormalVector3fMap() = pointnormals->points[i].getNormalVector3fMap();
    }
}


int main(int argc, char** argv)
{
    //read 2 clouds
    std::string cloud_filename_1(argv[1]);
    std::string cloud_filename_2(argv[2]);
    float initial_angle(std::stof(argv[3]));

    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr input_cloud_1(new pcl::PointCloud<pcl::PointXYZIRCT>);
    if (pcl::io::loadPCDFile<pcl::PointXYZIRCT>(cloud_filename_1, *input_cloud_1) == -1)
    {
        std::cerr << "Cloud NOT load file: " << cloud_filename_1 << "\n";
        exit(1);
    }

    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr input_cloud_2(new pcl::PointCloud<pcl::PointXYZIRCT>);
    if (pcl::io::loadPCDFile<pcl::PointXYZIRCT>(cloud_filename_2, *input_cloud_2) == -1)
    {
        std::cerr << "Cloud NOT load file: " << cloud_filename_2 << "\n";
        exit(1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_cloud_2(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_cloud_1_ds(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_cloud_2_ds(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_cloud_1_ds_aligned(new pcl::PointCloud<pcl::PointXYZ>);

    extractTopAndFlatten(input_cloud_1, flat_cloud_1);
    extractTopAndFlatten(input_cloud_2, flat_cloud_2);

    pcl::VoxelGrid<pcl::PointXYZ> low_res_filter;
    low_res_filter.setLeafSize(0.2f, 0.2f, 0.2f);


    low_res_filter.setInputCloud(flat_cloud_1);
    low_res_filter.filter(*flat_cloud_1_ds);
    flat_cloud_1_ds_aligned = flat_cloud_1_ds;

    low_res_filter.setInputCloud(flat_cloud_2);
    low_res_filter.filter(*flat_cloud_2_ds);


    Eigen::AngleAxisd rot_yaw(initial_angle / 180.0f * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Matrix<double,4,4> initial_guess = Eigen::Matrix<double,4,4>::Identity();
    initial_guess.block<3,3>(0,0) = rot_yaw.toRotationMatrix();

    // 2d icp start
    // compute normals for 2d points
    // pcl::PointCloud<pcl::Normal>::Ptr normals_src (new pcl::PointCloud<pcl::Normal>);
    // pcl::PointCloud<pcl::Normal>::Ptr normals_tgt (new pcl::PointCloud<pcl::Normal>);
    
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src_aligned (new pcl::PointCloud<pcl::PointNormal>);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals_src, normals_tgt;
    addNormal(flat_cloud_1_ds, normals_src, points_with_normals_src);
    addNormal(flat_cloud_2_ds, normals_tgt, points_with_normals_tgt);
    // addNormal(flat_cloud_1_ds_aligned, points_with_normals_src_aligned);

    std::cout << "starting icp... \n";
    auto time_start = std::chrono::system_clock::now();
    IcpAlignResult result_coarse = performCoarseIcp(
            points_with_normals_src,
            points_with_normals_tgt,
            points_with_normals_src_aligned,
            initial_guess.cast<float>());
    auto time_end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    std::cout << "ICP time cost: " << duration.count() * 1e-3 << "ms. \n" << endl;
    std::cout << "is icp converged: " << result_coarse.is_converged <<
                 ", fitness score: " << result_coarse.fitness_score <<
                 ", trans: \n" << result_coarse.final_transformation << ". \n";
    // 2d icp end


    // 3d icp start
    pcl::VoxelGrid<pcl::PointXYZIRCT> full_low_res_filter;
    full_low_res_filter.setLeafSize(0.2f, 0.2f, 0.2f);

    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr full_cloud_1_ds(new pcl::PointCloud<pcl::PointXYZIRCT>);
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr full_cloud_2_ds(new pcl::PointCloud<pcl::PointXYZIRCT>);
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr full_cloud_1_ds_aligned(new pcl::PointCloud<pcl::PointXYZIRCT>);

    full_low_res_filter.setInputCloud(input_cloud_1);
    full_low_res_filter.filter(*full_cloud_1_ds);

    full_low_res_filter.setInputCloud(input_cloud_2);
    full_low_res_filter.filter(*full_cloud_2_ds);

    auto time_start_3d = std::chrono::system_clock::now();
    IcpAlignResult result_fine = performFineIcp(
            full_cloud_1_ds,
            full_cloud_2_ds,
            full_cloud_1_ds_aligned,
            result_coarse.final_transformation);

    auto time_end_3d = std::chrono::system_clock::now();
    auto duration_3d = std::chrono::duration_cast<std::chrono::microseconds>(time_end_3d - time_start_3d);
    cout << "ICP time cost: " << duration_3d.count() * 1e-3 << "ms. \n" << endl;
    std::cout << "is icp converged: " << result_fine.is_converged <<
              ", fitness score: " << result_fine.fitness_score <<
              ", trans: \n" << result_fine.final_transformation << ". \n";
    // 3d icp end

    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_pc_1_pnts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr flat_pc_1_nmls(new pcl::PointCloud<pcl::Normal>);
    separatePointsAndNormals(points_with_normals_src, flat_pc_1_pnts, flat_pc_1_nmls);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer")); //创建视窗对象，定义标题栏名称“3D Viewer”
	viewer->addPointCloud<pcl::PointXYZ>(flat_pc_1_pnts, "original_cloud");	//将点云添加到视窗对象中，并定义一个唯一的ID“original_cloud”
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0.5, "original_cloud"); //点云附色，三个字段，每个字段范围0-1
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::PointNormal>(flat_pc_1_pnts, points_with_normals_src, 5, 1, "normals");	//每十个点显示一个法线，长度为0.05

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}

    /*
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->initCameraParameters();

    int v1(0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr flat_pc_1_pnts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr flat_pc_1_nmls(new pcl::PointCloud<pcl::Normal>);
    separatePointsAndNormals(points_with_normals_src, flat_pc_1_pnts, flat_pc_1_nmls);

    viewer->createViewPort(0.0, 0.0, 0.33, 0.5, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_1(flat_pc_1_pnts, 255, 0, 0);
    pcl::io::savePCDFileBinary("../flat_cloud_1_ds.pcd", *flat_pc_1_pnts);
    viewer->addPointCloud(flat_pc_1_pnts, color_handler_1, "cloud_1", v1);
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(flat_pc_1_pnts, flat_pc_1_nmls, 10.0, 10.0, "cloud_1_norm", v1);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_1_norm", v1);

    int v2(1);
    viewer->createViewPort(0.33, 0.0, 0.66, 0.5, v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_2(flat_cloud_2_ds, 0, 0, 255);
    pcl::io::savePCDFileBinary("../flat_cloud_2_ds.pcd", *flat_cloud_2_ds);
    viewer->addPointCloud(flat_cloud_2_ds, color_handler_2, "cloud_2", v2);

    int v3(2);
    viewer->createViewPort(0.66, 0.0, 1.0, 0.5, v3);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_handler_3(points_with_normals_src_aligned, 100, 0, 255);
    pcl::io::savePCDFileBinary("../flat_cloud_1_ds_aligned.pcd", *points_with_normals_src_aligned);
    viewer->addPointCloud(points_with_normals_src_aligned, color_handler_3, "cloud_3", v3);


    // 3d clouds
    int v4(3);
    viewer->createViewPort(0.0, 0.5, 0.33, 1.0, v4);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZIRCT> color_handler_4(full_cloud_1_ds, 255, 0, 0);
    pcl::io::savePCDFileBinary("../full_cloud_1_ds.pcd", *full_cloud_1_ds);
    viewer->addPointCloud(full_cloud_1_ds, color_handler_4, "cloud_1_ds", v4);

    int v5(4);
    viewer->createViewPort(0.33, 0.5, 0.66, 1.0, v5);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZIRCT> color_handler_5(full_cloud_2_ds, 0, 0, 255);
    pcl::io::savePCDFileBinary("../full_cloud_2_ds.pcd", *full_cloud_2_ds);
    viewer->addPointCloud(full_cloud_2_ds, color_handler_5, "cloud_2_ds", v5);

    int v6(5);
    viewer->createViewPort(0.66, 0.5, 1.0, 1.0, v6);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZIRCT> color_handler_6(full_cloud_1_ds_aligned, 100, 0, 255);
    pcl::io::savePCDFileBinary("../full_cloud_1_ds_aligned.pcd", *full_cloud_1_ds_aligned);
    viewer->addPointCloud(full_cloud_1_ds_aligned, color_handler_6, "cloud_1_ds_aligned", v6);

    viewer->setBackgroundColor(1, 1, 1, v1);
    viewer->setBackgroundColor(1, 1, 1, v2);
    viewer->setBackgroundColor(1, 1, 1, v3);
    viewer->setBackgroundColor(1, 1, 1, v4);
    viewer->setBackgroundColor(1, 1, 1, v5);
    viewer->setBackgroundColor(1, 1, 1, v6);
    viewer->addCoordinateSystem();
    viewer->spin();


    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);

    }
    */

    return 0;
}