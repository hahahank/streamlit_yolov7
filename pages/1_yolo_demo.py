import streamlit as st
import time
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import numpy as np
import torch
import cv2
import random
import queue
# 載入YOLOv7模型
from models.experimental import attempt_load
from utils.general import non_max_suppression
from  hubconf import custom
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device",device)
my_w="weights/yolov7.pt"
model = custom(path_or_model=my_w)  # custom example
model.iou = 0.9
model.conf = 0.5
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

st.set_page_config(page_title="Yolo Demo", page_icon="📈")
st.sidebar.header("Yolo Demo")

#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)

conf_threshold = st.slider("Conf", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

# Radom Collor
CLASS_COLORS = {}
def get_color_for_category(category):
    global CLASS_COLORS
    if category not in CLASS_COLORS:
        # 随机生成一个种子，以便每次运行时使用不同的颜色分布
        random.seed(category)
        # 通过哈希函数生成红、绿、蓝三种颜色的分量
        r = random.randint(30, 220)
        g = random.randint(30, 220)
        b = random.randint(30, 220)
        color = (r, g, b)
        CLASS_COLORS[category] = color
    else:
        color = CLASS_COLORS[category]
    return color

def yolo_detection(frame: av.VideoFrame) -> av.VideoFrame:
    model.conf = conf_threshold
    # Convert VideoFrame to numpy ndarray
    img = frame.to_ndarray(format="bgr24")

    # Run object detection using yolov5
    results = model(img)
    print("RESULT = ",type(results), results)
    df_prediction = results.pandas().xyxy
    print("results.pandas():",type(results.pandas().xyxy))
    
    for result in df_prediction:
        print("result:",type(result))
        result = result.to_dict(orient='records')
        for i in  result:
            xmin = int(i.get("xmin"))
            xmax = int(i.get("xmax"))
            ymin = int(i.get("ymin"))
            ymax = int(i.get("ymax"))
            class_id = i.get("class")
            class_name = i.get("name")
            confidence = i.get("confidence")
            class_color = get_color_for_category(class_id)
            print(xmin,xmax,ymin,ymax,class_id,class_name,confidence)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),class_color, 2)
            cv2.putText(img, "({0}){1} : {2:.2f}".format(class_id,class_name,confidence), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
        result_queue.put(result)	# Detectioon Queue
    return av.VideoFrame.from_ndarray(img, format="bgr24")
    #return img

def runSnapshot():
    print("Snapshot")
    print("Snapshot")
    print("Snapshot")
    print("Snapshot")



webrtc_ctx = webrtc_streamer(
    key="yolo-detection",
    mode=WebRtcMode.SENDRECV,
#    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=yolo_detection,
    media_stream_constraints={"video": True, "audio": False},
 #   async_processing=True,
)


if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
