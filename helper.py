from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import os

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def save_uploaded_video(uploaded_video):
        save_path = os.path.join("videos", uploaded_video.name)
        print(save_path)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success(f"Uploaded video saved to {save_path}")

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracking algorithm", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    output_fram = res[0].plot()
    st_frame.image(output_fram,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    return output_fram


def frame_detection(video_details,conf, model, is_display_tracker, tracker):
    try:
        frames=[]
        print(f"detecting : ",{video_details["path"]})
        vid_cap = cv2.VideoCapture(video_details["path"])
        video_writer = cv2.VideoWriter_fourcc(*'MJPG')
        out = None
        st_frame = st.empty()

        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if not success:
                break  # Exit loop if no more frames

            # Process frame and detect objects
            output_fram = _display_detected_frames(conf, model, st_frame, frame, is_display_tracker, tracker)
            frames.append(output_fram)

            # Initialize the VideoWriter once we know the frame size
            if out is None:
                height, width, _ = output_fram.shape
                output_path = "output_video.avi"  # Define the output path
                out = cv2.VideoWriter(output_path, video_writer, 30.0, (width, height))
            
            # Write the processed frame to the output video
            out.write(output_fram)
            print("Saving frame...")
        
        # Release resources
        vid_cap.release()
        if out is not None:
            out.release()
        
        # Display the video in Streamlit
        st.write("Processed video:")
        
        #viewing the whole video after being processed by uolo
        # with open(output_path, 'rb') as output_video_file:
        #     video_bytes = output_video_file.read()
        # output_video_details={"bytes":video_bytes , "path":os.path.join(os.getcwd(),output_path)}
        # print(output_video_details["path"])
        # st.video(output_video_details["bytes"])
            
    except Exception as e:
        st.sidebar.error("Error loading video: " + str(e))

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.
    """
    
    default_source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    
    try:
        # File uploader widget
        is_display_tracker, tracker = display_tracker_options()
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            video_bytes = uploaded_video.read()
            save_uploaded_video(uploaded_video)
            video_details={"bytes":video_bytes , "path":f"videos/{uploaded_video.name}"}
            print(f"selecting :",video_details["path"])

        else :
            with open(settings.VIDEOS_DICT.get(default_source_vid), 'rb') as default_video_file:
                video_bytes = default_video_file.read()
            video_details={"bytes":video_bytes , "path":settings.VIDEOS_DICT.get(default_source_vid)}
            print(f"selecting :",video_details["path"])

        st.video(video_details["bytes"])

    except Exception as e:
        st.sidebar.error("Error uploading video: " + str(e))

    if st.sidebar.button('Detect Video Objects'):
        frame_detection(video_details,conf, model, is_display_tracker, tracker)