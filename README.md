# Using YOLOv8 in ROS melodic

## Environment
**python3.6.9**    
**torch-1.7.1+cu110**

## How to run
<p style="text-align: justify;">

**1.** Download the code

    git clone https://github.com/fbh16/yolov8_ros.git  
    cd yolov8_ros/src/yolov8_ros
    git clone https://github.com/ultralytics/ultralytics.git
    cd .. && catkin_make
    source devel/setup.bash

**2.** Modify the parameters in the **launch file**, such as the **model path**, **ROS topics**, **visualization options**, to suit your own requirements. 

**2.** Modify the parameters in the **launch file**, such as the **model path**, **ROS topics**, **visualization options**, to suit your own requirements, download the YOLOv8 pretrained model at <https://github.com/ultralytics/assets/releases>. 

**3.** 
If you are using a **Python version lower than 3.8**, modify the code at **line 76** of the file(**ultralytics/ultralytics/hub/auth.py**):  

    if header := self.get_auth_header():  

to:  

    header = self.get_auth_header()
    if header:

**3.** Start the camera node or the rosbag, check the topics, and run the launch file.

    roslaunch yolov8_ros v8.launch
    
</p>

## Parameters
<p style="text-align: justify;">

**device:** Type 'cpu' if you are using CPU, and 'cuda' if you are using GPU.When you need to use CPU, you'll need to install a Python package with:   
`python3 -m pip install py-cpuinfo`

**model:** Enter the absolute path to the YOLOv8 model on your local machine.

**class_yaml:** Please provide the specific path to the YAML file associated with your dataset. The model can obtain the classes of the dataset from this YAML file.

**view_image:** Whether to show detection screen.

**hide_label:** Whether to hide the BoundingBox label.

**sub&pub:** ROS publish and subscribe topics.

**single_class:** When you want to detect all classes, you don't need to input anything. When you want to display only a specific class, enter the name of that class, i.e., 'person'.  

Some parameters, such as input image size and intersection over union (IoU), have default values set but can be modified as needed.
