import streamlit as st
import cv2
import numpy as np
from PIL import Image
from emotion_detection_from_image import inference
from emotion_detection_from_video import process_video


def main():
    st.title("Emotion Detection")
    
    option = st.radio("Çalıştırma şeklinizi seçin:", ("Resim", "Video"))
    
    if option == "Resim":
        uploaded_file = st.file_uploader("Resim seçin", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            result = inference(image)
            
            st.image(result, caption="Duygu tespiti sonucu", use_column_width=True)
    
    elif option == "Video":
        st.warning("Video işleniyor...")
        process_video()
        
if __name__ == "__main__":
    main()
    
    
   