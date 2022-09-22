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
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>

using namespace std;
//typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

struct PointXYZIRCL
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring_row;
    uint16_t ring_col;
    int32_t  label_id;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (
	PointXYZIRCL,
	(float, x, x)
	(float, y, y)
	(float, z, z)
	(float, intensity, intensity)
	(uint16_t, ring_row, ring_row)
	(uint16_t, ring_col, ring_col)
	(int32_t,  label_id, label_id)
)

// typedef pcl::PointXYZI PointType;
typedef PointXYZIRCL   PointType;

int main(int argc,char **argv)
{
	pcl::PointCloud<PointType>::Ptr cloud_input(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud_flat(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
	pcl::PointCloud<NormalType>::Ptr cloud_normal(new pcl::PointCloud<NormalType>());

	std::string fileName(argv[1]);
	pcl::io::loadPCDFile(fileName, *cloud_input);

    for (int i = 0; i < (int)cloud_input->size(); i ++) {
        PointType po;
        PointType &pi = cloud_input->points[i];

        float range = sqrt(pi.x * pi.x + pi.y * pi.y);

        if (pi.z < 0.0f || range > 30.0f || pi.label_id <= 0) {
            continue;
        }

        po = pi;
        //cloud->push_back(po);
        po.z = 0.0f;
        cloud->push_back(po);
    }

    std::cout << "cloud_in: " << cloud_input->size() << ", filter: " << cloud->size() << std::endl;

    //compute mean and covariance of the cloud
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);

    //compute eigen values ands vectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	//eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

	Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
	transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
	transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3,3>(0,0)) * (pcaCentroid.head<3>());//

	pcl::PointCloud<PointType>::Ptr transformedCloud(new pcl::PointCloud<PointType>);
	pcl::transformPointCloud(*cloud, *transformedCloud, transform);

	std::cout << eigenValuesPCA << std::endl;
	std::cout << eigenVectorsPCA << std::endl;


	//初始位置时的主方向
	PointType c;
	c.x = pcaCentroid(0);
	c.y = pcaCentroid(1);
	c.z = pcaCentroid(2);
	PointType pcZ;
	pcZ.x = 200 * eigenVectorsPCA(0, 0) + c.x;
	pcZ.y = 200 * eigenVectorsPCA(1, 0) + c.y;
	pcZ.z = 200 * eigenVectorsPCA(2, 0) + c.z;
	PointType pcY;
	pcY.x = 200 * eigenVectorsPCA(0, 1) + c.x;
	pcY.y = 200 * eigenVectorsPCA(1, 1) + c.y;
	pcY.z = 200 * eigenVectorsPCA(2, 1) + c.z;
	PointType pcX;
	pcX.x = 200 * eigenVectorsPCA(0, 2) + c.x;
	pcX.y = 200 * eigenVectorsPCA(1, 2) + c.y;
	pcX.z = 200 * eigenVectorsPCA(2, 2) + c.z;
       //visualization
	pcl::visualization::PCLVisualizer viewer;
	pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler(cloud,255, 0, 0);
	viewer.addPointCloud(cloud,color_handler,"cloud");


	viewer.addArrow(pcZ, c, 0.0, 0.0, 1.0, false, "arrow_z");
	viewer.addArrow(pcY, c, 0.0, 1.0, 0.0, false, "arrow_y");
	viewer.addArrow(pcX, c, 1.0, 0.0, 0.0, false, "arrow_x");


	viewer.addCoordinateSystem(100);
	viewer.setBackgroundColor(1.0, 1.0, 1.0);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);

	}

	return 0;
}
