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
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    return res_plotted


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
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
        try:
            print(f"detecting : ",{video_details["path"]})
            vid_cap = cv2.VideoCapture(video_details["path"])
            video_writer = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            
            
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    res_plotted=_display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                    
                    if out is None:
                        # Initialize the video writer
                        height, width, _ = res_plotted.shape
                        out = cv2.VideoWriter('.', video_writer, 30.0, (width, height))
                    
                    # Write the processed frame to the output video
                    out.write(res_plotted)
                    print("saving .........frame")
                             
                                
                else:
                    vid_cap.release()
                    out.release()
                    break
                
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))