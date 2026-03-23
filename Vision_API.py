## Import Bin
import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw
import random
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


st.set_page_config(page_title="Vision API")

t1, t2, t3, t4=st.tabs(["🔍 Object Detection","📷 Live Object Detection","",""])


@st.cache_resource
def load_model():
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector=load_model()


colors={}
def get_color(label):
    if label not in colors:
        colors[label]=(
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return colors[label]


with t1:

    st.title("🔍 Object Detection")
    st.write("Upload an image and the AI will detect objects in it!")

    uploaded_file=st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    threshold=st.slider("Confidence Threshold (Higher -> Accurate Prediction)", 0.1, 1.0, 0.7, 0.05, key="t1_threshold")

    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        with st.spinner("Detecting objects..."):
            results=detector(image, threshold=threshold)

        draw=ImageDraw.Draw(image)

        for result in results:
            label=result["label"]
            score=result["score"]
            box=result["box"]
            color=get_color(label)

            draw.rectangle(
                [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                outline=color,
                width=3,
            )

            text=f"{label} {score:.0%}"
            draw.rectangle(
                [box["xmin"], box["ymin"] - 18, box["xmin"] + len(text) * 7, box["ymin"]],
                fill=color,
            )
            draw.text((box["xmin"] + 2, box["ymin"] - 16), text, fill="white")

        st.subheader(f"Detected Image ({len(results)} objects found)")
        st.image(image, use_container_width=True)

        st.subheader("Detection Results")
        for r in results:
            st.write(f"✅ **{r['label']}** — confidence: `{r['score']:.2%}`")


with t2:

    st.title("📷 Live Object Detection")
    st.write("Allow camera access and the AI will detect objects in real time!")

    live_threshold=st.slider("Confidence Threshold (Higher -> Accurate Prediction)", 0.1, 1.0, 0.5, 0.05, key="t2_threshold")

    class ObjectDetectionProcessor(VideoProcessorBase):

        def recv(self, frame):
            img=frame.to_image()

            results=detector(img, threshold=live_threshold)

            draw=ImageDraw.Draw(img)

            for result in results:
                label=result["label"]
                score=result["score"]
                box=result["box"]
                color=get_color(label)

                draw.rectangle(
                    [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                    outline=color,
                    width=3,
                )

                text=f"{label} {score:.0%}"
                draw.rectangle(
                    [box["xmin"], box["ymin"] - 18, box["xmin"] + len(text) * 7, box["ymin"]],
                    fill=color,
                )
                draw.text((box["xmin"] + 2, box["ymin"] - 16), text, fill="white")

            return av.VideoFrame.from_image(img)

    webrtc_streamer(
        key="live-object-detection",
        video_processor_factory=ObjectDetectionProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    )