#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from std_msgs.msg import Header

class GridMapNode(Node):
    def __init__(self):
        super().__init__('grid_map_node')

        # 参数，可根据需求调整
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('voxel_size', 0.05)  # 体素滤波大小
        self.declare_parameter('inflation_radius', 0.2)  # 点云膨胀半径

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value

        # 订阅原始点云
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/odin1/cloud_raw',
            self.cloud_callback,
            10
        )

        # 发布处理后的点云
        self.cloud_pub = self.create_publisher(PointCloud2, '/grid_map/cloud_processed', 10)

        self.get_logger().info("GridMapNode initialized.")

    def cloud_callback(self, msg: PointCloud2):
        # 将 ROS PointCloud2 转为 numpy array
        cloud_points = []
        for p in pc2.read_points(msg, skip_nans=True):
            cloud_points.append([p[0], p[1], p[2]])
        if len(cloud_points) == 0:
            return
        np_cloud = np.array(cloud_points, dtype=np.float32)

        # 使用 Open3D 进行处理
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud)

        # 1. 体素滤波下采样
        o3d_cloud = o3d_cloud.voxel_down_sample(self.voxel_size)

        # 2. 点云膨胀（通过球体膨胀）
        inflated_points = []
        radius = self.inflation_radius
        for pt in np.asarray(o3d_cloud.points):
            # 在每个轴向上生成膨胀点
            x = np.arange(pt[0] - radius, pt[0] + radius + self.voxel_size, self.voxel_size)
            y = np.arange(pt[1] - radius, pt[1] + radius + self.voxel_size, self.voxel_size)
            z = np.arange(pt[2] - radius, pt[2] + radius + self.voxel_size, self.voxel_size)
            for xi in x:
                for yi in y:
                    for zi in z:
                        inflated_points.append([xi, yi, zi])
        if len(inflated_points) == 0:
            return

        inflated_cloud = o3d.geometry.PointCloud()
        inflated_cloud.points = o3d.utility.Vector3dVector(np.array(inflated_points, dtype=np.float32))

        # 转回 ROS PointCloud2
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        ros_msg = pc2.create_cloud_xyz32(header, np.asarray(inflated_cloud.points).tolist())

        # 发布
        self.cloud_pub.publish(ros_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GridMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
