# -FRE--Flower-Detection-and-Location
# Task Description
In this task, the field robots have to work in a grassland area (see Figure Below). The task is to find and map any weeds in the area. In order to support the localisation within the grassland area, it will be enclosed by a side fence, e.g. straw bales or something similar with a height of at least 0.3 m. The zero position of the coordinate system will be marked by a vertical board or wooden plank. The result data consisting of (x,y) data sets has to be provided by the field robot. The position has to be sent as soon as the weed has been detected. The weed will be represented by small artificial flowers (https://www.amazon.de/-/en/artificial-sunflowers-flowers-decorative-decoration/dp/B08SWGFLQS/ref=sr_1_2?crid=3TS5QB5RDBNUP&th=1) placed on the ground. The flowers may have any non-green VIS color. The dimensions of the quadratic grassland area are approx. 8 m x 8 m. The distance between the following objects is 1.0 m.

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/4f8aaa08-a3f9-4d73-af41-59b659a1aea4" />


# Sensor Used
Realsense D435i Camera

# Pre-requisites
Install ROS2 Humble: https://docs.ros.org/en/humble/index.html

Install Realsense ROS2 Wrapper: https://github.com/IntelRealSense/realsense-ros

# Steps of Implementation
Clone the repository
```bash
git clone https://github.com/swarajtendulkar10/-FRE--Flower-Detection-and-Location
```

Add folders to the root folder


Build the workspace
```bash
colcon build
```

Run the Detection Node

