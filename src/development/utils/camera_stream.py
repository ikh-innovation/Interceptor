import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()

        # SUBSCRIBE TO CAMERA
        self.subscription = self.create_subscription(
            Image,
            '/World/quadrotor/Camera', #CHANGE THIS based on Camera name
            self.image_callback,
            10
        )

        # VIDEO WRITER (initialized after first frame)
        self.video_writer = None
        self.fps = 30  # adjust if needed

        self.get_logger().info("Camera subscriber started")

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Initialize VideoWriter once we know frame size
        if self.video_writer is None:
            height, width, _ = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            self.video_writer = cv2.VideoWriter(
                'output.mp4',
                fourcc,
                self.fps,
                (width, height)
            )

            self.get_logger().info(f"Recording video: {width}x{height} @ {self.fps} FPS")

        # Write frame to video
        self.video_writer.write(frame)

        # Display
        cv2.imshow("Drone Camera", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        # Release video file properly
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info("Video saved successfully")

        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()

    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()