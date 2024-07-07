import streamlit as st
from PIL import Image
import numpy as np
from models.model import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(1, 62)
model = torch.load('weights/emnist-resnet18-full.pth', map_location=device)
model = model.to(device)
model.eval()

st.set_page_config(
    page_title="OCR Handwrite Digit Recognition",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("✍️ OCR Handwrite Digit Recognition")
st.markdown("Upload an image of handwritten digits to recognize them using a pre-trained model.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil.convert('RGB'))
    st.image(image_np, caption='Uploaded Image.', use_column_width=True)
    if st.button('Predict'):  # Sử dụng button để kích hoạt sự kiện predict
        st.write("")
        predictions,final_predict = predict_sequence(model, image_np,device)
        digit = ''.join([key[0][0] for key in predictions])
        score = np.mean([key[0][1] for key in predictions])
        st.write(f"Recognized Digit: {digit}")
        st.write(f"Score: {score}")
# Thêm icon và thông tin cho ứng dụng
st.sidebar.markdown("### About")
st.sidebar.markdown("This app recognizes handwritten digits using a pre-trained OCR model.")
st.sidebar.image("weight/icon.png", use_column_width=True)
