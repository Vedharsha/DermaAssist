# DermaAssist
# DERMATOLOGICAL DISEASE DETECTION AND ENVIRONMENT-BASED  SKIN HEALTH ASSISTANCE

Overview

The Dermatological Disease Detection and Environment-Based Skin Health Assistance System is an intelligent web-based solution designed to enable early detection and effective management of skin diseases using deep learning.

The system uses a Convolutional Neural Network (CNN) based on EfficientNet-B1 to accurately classify 32 different skin diseases from input images. Unlike traditional systems that rely only on image analysis, this solution improves diagnostic accuracy by integrating user-reported symptoms and real-time environmental data such as temperature, humidity, and UV index.

A dedicated severity analysis module evaluates symptoms like itching, redness, and swelling to classify conditions into mild, moderate, or severe categories. Additionally, the environmental module identifies external factors that may trigger or worsen the condition.

By combining all these inputs, the system provides personalized skincare recommendations, preventive measures, and guidance on when to seek medical attention.

This project aims to reduce delayed diagnosis, increase awareness, and provide accessible dermatological support, especially in remote and underserved areas.

## Training (`train_model.py`) — Kaggle Notebook (GPU T4×2)

1. Go to [kaggle.com](https://www.kaggle.com) → **Create Notebook** → Upload `train_model.py`
2. Set **Accelerator** → `GPU T4×2` in the right sidebar
3. Add the dataset: search and add `skindiseasedataset`
4. Add this at the top of the notebook and run:
   ```python
   !pip install albumentations opencv-python-headless
   ```
5. Click **Run All**
6. Download the output `trained_model.pth` from `/kaggle/working/`

---

## Running the App (`main.py`) — VSCode (Python 3.10.0)

Place these files in the same folder:
```
main.py
trained_model.pth          ← downloaded from Kaggle
disease_symptoms.json
disease_recommendations.json
```

Then in the terminal:
```bash
pip install -r requirements.txt
streamlit run main.py
```

App opens at `http://localhost:8501`


#  Experiment Results
# Step 1: Upload image, Disease classification
<img width="1902" height="923" alt="image" src="https://github.com/user-attachments/assets/f5122908-35e5-4012-9963-cccd02e47516" />

# Step 2: Disease-based symptom questionnaires and personalised questionnaires
<img width="1888" height="969" alt="image" src="https://github.com/user-attachments/assets/075d6558-7632-4dc2-8a69-7c0d2a79ada5" />

# Step 3: Severity Level, Personalised Recommendations
# Case 1:
<img width="1862" height="926" alt="image" src="https://github.com/user-attachments/assets/7948ea7a-5d44-411f-be2b-04c9e51f5022" />

# Case 2: Unseen Conditions
<img width="1500" height="687" alt="image" src="https://github.com/user-attachments/assets/c0878f39-2955-45c4-9977-0b6faf19deaf" />
