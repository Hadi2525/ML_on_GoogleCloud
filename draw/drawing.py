import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

model = keras.models.load_model("nn.h5")

# Specify canvas parameters in application
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )

# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

# realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# # Create a canvas component
# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,
#     drawing_mode=drawing_mode,
#     point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
#     key="canvas",
# )

# # Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)

data = st_canvas(update_streamlit=False, 
                 key="canvas",
                 width=100,
                 height=100,
                 stroke_width=2,
                 background_color=(0,0,0),
                 stroke_color='white')
if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA").convert('L')
        data = np.asarray(im)
        data = data / 255.0
        data = data[np.newaxis, ..., np.newaxis]
        # --> [1, x, y, 1]

        data = tf.image.resize(data, [28, 28])
        predictions = model(data)
        predictions = tf.nn.softmax(predictions)
        pred0 = predictions[0]
        label0 = np.argmax(pred0)
        print(label0)