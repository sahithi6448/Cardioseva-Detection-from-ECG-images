import streamlit as st
from skimage.io import imread
from skimage import color
from skimage.transform import resize
from skimage.filters import gaussian, threshold_otsu
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib

st.title("CardioSeva – ECG Disease Detection")

# -------------------------------
# 📌 1. Upload ECG Image
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload your ECG image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Display image
    image = imread(uploaded_file)
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)
    st.success("✅ ECG Image successfully uploaded!")

    # ---------------------------------------------------
    # 📌 2. Convert to Gray & Resize for Processing
    # ---------------------------------------------------
    image_gray = color.rgb2gray(resize(image, (1572, 2213)))

    # ---------------------------------------------------
    # 📌 3. Apply Gaussian Blur & Otsu Thresholding
    # ---------------------------------------------------
    blurred = gaussian(image_gray, sigma=0.8)
    thresh = threshold_otsu(blurred)
    binary = blurred < thresh

    # ---------------------------------------------------
    # 📌 4. Extract ECG Contour
    # ---------------------------------------------------
    contours = measure.find_contours(binary, 0.8)
    
    if len(contours) == 0:
        st.error("❌ No ECG signal detected. Please try another image.")
        st.stop()

    # Select the longest contour
    main_contour = sorted(contours, key=lambda x: x.shape[0], reverse=True)[0]

    # Resize contour to uniform size
    signal = resize(main_contour, (255, 2))

    # ---------------------------------------------------
    # 📌 5. Normalize Signal using MinMaxScaler
    # ---------------------------------------------------
    scaler = MinMaxScaler()
    scaled_signal = scaler.fit_transform(signal)

    # X_test expects one row
    X_test = pd.DataFrame(scaled_signal[:, 0]).T

    # ---------------------------------------------------
    # 📌 6. Load Trained Model
    # ---------------------------------------------------
    try:
        model = joblib.load("Heart_Disease_Prediction_using_ECG.pkl")
    except:
        st.error("❌ Model file not found! Please place 'Heart_Disease_Prediction_using_ECG.pkl' in the project folder.")
        st.stop()

    # ---------------------------------------------------
    # 📌 7. Predict Disease
    # ---------------------------------------------------
    prediction = model.predict(X_test)[0]

    # Label mapping
    labels = {
        0: "Myocardial Infarction (MI)",
        1: "Abnormal Heartbeat",
        2: "Normal ECG",
        3: "History of MI"
    }

    # ---------------------------------------------------
    # 📌 8. Display Prediction
    # ---------------------------------------------------
    st.subheader("🩺 Predicted Disease:")
    st.success(f"{labels[prediction]}")
