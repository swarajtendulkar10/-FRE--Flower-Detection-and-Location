#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import math

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/your_workspace/src/obj_track/src/flower.pt', force_reload=True)

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Write your YOLOv5 depth scale here
depth_scale = 0.001

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'detections_publisher', 10)
        self.subscription = self.create_subscription(Odometry,'/odometry/transformed',self.listener_callback,10)
        self.i = 0
        self.f = 222.25
        self.g=546.5
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.xi = 0.0
        self.yi = 0.0
        self.angle=0.0
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.xt=25557.0
        self.yt=5257.0
        self.a=0
        self.subscription

    def listener_callback(self, msg):
        self.xi=msg.pose.pose.position.x
        self.yi=msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.angle = yaw
        self.a=math.degrees(yaw)
        print(f'angle: {self.angle}')

    def timer_callback(self):
        msg = String()

        # Get the latest frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert the frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Convert the depth image to meters
        depth_image = depth_image * depth_scale

        # Detect objects using YOLOv5
        results = model(color_image)

        # Process the results
        for result in results.xyxy[0]:
            x1, y1, x2, y2, confidence, class_id = result
            if confidence > 0.85:
                # Calculate the distance to the object
                object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])
                object_depth=np.sqrt((object_depth**2)-(.33**2)) + 0.34
                if 0.0<self.a<90.0 and object_depth!=0:
                    xo=object_depth*np.sin(self.angle)
                    yo=object_depth*np.cos(self.angle)
                    self.xt=self.xi-xo
                    self.yt=self.yi+yo


                if 90.0<self.a<180.0 and object_depth!=0:
                    xo=object_depth*np.cos(self.angle-90)
                    yo=object_depth*np.sin(self.angle-90)
                    self.xt=self.xi-xo
                    self.yt=self.yi-yo

                if -179.0<self.a<-90.0 and object_depth!=0:
                    xo=object_depth*np.sin(self.angle)
                    yo=object_depth*np.cos(self.angle)
                    self.xt=self.xi+xo
                    self.yt=self.yi-yo

                if -90.0<self.a<0.0 and object_depth!=0:
                    xo=object_depth*np.cos(self.angle+90)
                    yo=object_depth*np.sin(self.angle+90)
                    self.xt=self.xi+xo
                    self.yt=self.yi+yo

                x_centers = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                x_center = 320 - x_centers
                pixel_size_meters = 0.00155
                x_center = x_center * pixel_size_meters
                label = f"({float(x_center):.3f}): {object_depth:.2f}m"
                if x_center<0:
                    self.xt=self.xt+x_center
                else:
                    self.xt=self.xt-x_center


                # Draw a rectangle around the object
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)
                cv2.circle(color_image, (int(x_centers), int(y_center)), 5, (0, 0, 255), -1)

                # Draw the bounding box with label
                cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)
                
                if not self.xt-0.4 <= self.f <= self.xt+0.4 and not self.yt-0.4 <= self.g <= self.yt+0.4:
                    self.f = self.xt
                    self.g = self.yt
                    msg.data = f"{float(self.xt):.3f}+{float(self.yt):.3f}"
                    self.publisher_.publish(msg)

                # Print the object's class and distance
                print(f"{model.names[int(class_id)]}: {label}")

        # Show the image
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
