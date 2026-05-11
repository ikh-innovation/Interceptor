import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class CameraSubscriber(Node):

    def __init__(self):

        super().__init__('camera_subscriber')

        # CREATE BRIDGE
        self.bridge = CvBridge()

        # SUBSCRIBE TO CAMERA
        self.subscription = self.create_subscription(
            Image,
            '/World/quadrotor/body/camera',
            self.image_callback,
            10
        )

    def image_callback(self, msg):

        # ROS2 IMAGE → OpenCV IMAGE
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv2.imshow("Drone Camera", frame)

        cv2.waitKey(1)


def main():

    rclpy.init()

    node = CameraSubscriber()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()