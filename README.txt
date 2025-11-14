❤️ CardioSeva – ECG Disease Detection

CardioSeva is a lightweight ECG disease detection system that analyzes uploaded ECG images, extracts the waveform, and predicts heart conditions using a machine-learning ensemble model. Built with Streamlit, it provides a clean and user-friendly interface for quick diagnosis support.


🚀 Features

Upload ECG images (JPG/PNG)

Automatic contour extraction

Signal preprocessing (smoothing, thresholding, normalization)

Soft Voting Classifier for accurate predictions

Detects: Normal, MI, Abnormal Heartbeat, History of MI


⚙️ Run the App
pip install -r requirements.txt
streamlit run app.py



🧠 Model

A Soft Voting Classifier combining:
KNN
Logistic Regression
SVM
Random Forest
Gaussian Naive Bayes


🩺 Supported Outputs

✔ Myocardial Infarction (MI)
✔ Abnormal Heartbeat
✔ Normal ECG
✔ History of MI