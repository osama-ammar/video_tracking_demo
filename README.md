# <span style="color:deepskyblue"> Real-time Object Detection and Tracking with YOLOv8 </span>

This repository is an extensive open-source project showcasing the seamless integration of **object detection and tracking** using **YOLOv8** (object detection algorithm), along with **Streamlit** (a popular Python web application framework for creating interactive web apps). The project offers a user-friendly and customizable interface designed to detect and track objects in real-time video streams.
- try it  <https://videotrackingdemo-jzen4azltvfdudnjtb3nk4.streamlit.app/>

## Requirements

Python 3.6+
YOLOv8
Streamlit

```bash
pip install ultralytics streamlit pytube
```

## Installation

- Clone this repository
- Change to the repository directory: `cd yolov8-streamlit-detection-tracking`
- Create `weights`, `videos`, and `images` directories inside the project.
- Download the pre-trained YOLOv8 weights from (<https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt>) and save them to the `weights` directory in the same project.

## Usage

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.

### ML Model Config

- Select task (Detection, Segmentation)
- Select model confidence
- Use the slider to adjust the confidence threshold (25-100) for the model.

One the model config is done, select a source.


- Click on `Detect Video Objects` button and the selected task (detection/segmentation) will start on the selected video.


## Acknowledgements

This app uses [YOLOv8](<https://github.com/ultralytics/ultralytics>) for object detection algorithm and [Streamlit](<https://github.com/streamlit/streamlit>) library for the user interface.

### Disclaimer

Please note this project is intended for educational purposes only and should not be used in production environments.

**Hit star ⭐ if you like this repo!!!**
