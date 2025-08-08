import streamlit as st 
import numpy as np 
import requests
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image 
import google.generativeai as genai 
import joblib

genai.configure(api_key="AIzaSyBNeg_QMZh2fGE_bzbdEOfT5dktju9TF9E")  
model = genai.GenerativeModel(model_name="gemini-2.0-flash")
chat = model.start_chat(history=[])

ml = load_model("D:\\demo-covid\\Data\\covid_pneu_model.h5")
class_ = ["Covid19", "Normal", "Pneumonia"]

st.title("🩺 Pneumonia/COVID Diagnosis System")

upl = st.file_uploader("📷 Upload a  image", type=["png", "jpg", "jpeg"])

if upl:
    img = Image.open(upl).convert("RGB")
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    imgg = image.img_to_array(img)
    imgg = np.expand_dims(imgg, axis=0) / 255.0

    pred = ml.predict(imgg)
    ress = np.argmax(pred)
    pred_class = class_[ress]

    if pred_class == "Normal":
        st.success("✅ You're healthy!")
    else:
        st.warning(f"⚠ {pred_class} detected. Please provide additional health details.")
        st.sidebar.header("📋 Additional Patient Info")

        Age = st.sidebar.slider("Age", 0, 80)
        Gn = st.sidebar.selectbox("Gender", ["Male", "Female"])
        fv = st.sidebar.selectbox("Fever", ["Yes", "No"])
        cf = st.sidebar.selectbox("Cough", ["Yes", "No"])
        ft = st.sidebar.selectbox("Fatigue", ["Yes", "No"])
        brt = st.sidebar.selectbox("Breathlessness", ["Yes", "No"])
        cm = st.sidebar.selectbox("Comorbidity", ["Yes", "No"])
        stt = st.sidebar.selectbox("Stage", ["mild", "moderate", "severe"])
        tp = st.sidebar.selectbox("Type", ["viral", "bacteria"])
        ts = st.sidebar.slider("Tumor_Size", 0, 5)

        gn_num=1 if Gn=="Male" else 0
        fv_num=1 if fv=="Yes" else 0
        cf_num=1 if cf=="Yes" else 0
        ft_num=1 if ft=="Yes" else 0
        brt_num=1 if brt=="Yes"  else 0
        cm_num=1 if  cm=="Yes" else 0
        stt_num=0 if stt=="mild" else 1 if stt=="moderate" else 2
        tp_num=0 if tp=="viral" else 1

        data_inp={
            "Age":Age,
            "Gender":gn_num,
            "Fever":fv_num,
            "Cough":cf_num,
            "Fatigue":ft_num,
            "Breathlessness":brt_num,
            "Comorbidity":cm_num,
            "Stage":stt_num,
            "Type":tp_num,
            "Tumor_Size":ts
        }
        if st.sidebar.button("🔍 Predict "):
            try:
                res = requests.post("http://127.0.0.1:8000/prd", json=data_inp)
                response = res.json()
                severity = response.get("Prediction", 0)

                st.markdown(f"### 🔬 Predicted Severity Score: *{severity:.2f}*")
                if severity > 0.5:
                    st.error("⚠ High severity risk. Immediate attention recommended.")
                else:
                    st.success("🟢 Low severity. Monitor symptoms and stay alert.")

            except requests.exceptions.RequestException as e:
                st.error("❌ Could not connect to backend prediction API.")
                st.text(str(e))

st.markdown("---")
st.markdown("### 💬 Ask Questions...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for q, a in st.session_state.chat_history:
    st.markdown(f"❓ You:** {q}")
    st.markdown(f"🤖 Gemini:** {a}")
    st.markdown("---")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("🗨 Ask something...", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("Gemini is thinking..."):
        try:
            response = chat.send_message(user_input)
            st.session_state.chat_history.append((user_input, response.text))
        except Exception as e:
            st.session_state.chat_history.append((user_input, f"⚠ Error: {str(e)}"))

ml = joblib.load("comment.pkl")
vec = joblib.load("vector.pkl")
sent = {1: "Positive", 0: "Neutral", -1: "Negative"}

if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

if st.button("📝 Give Feedback"):
    st.session_state.show_feedback = not st.session_state.show_feedback

if st.session_state.show_feedback:
    st.sidebar.header("Feedback")
    feedback_text = st.sidebar.text_area("💬 Share your thoughts")

    if st.sidebar.button("Submit"):
        if feedback_text.strip() != "":
            v = vec.transform([feedback_text])
            prd = ml.predict(v)
            fb = sent.get(prd[0])
            st.sidebar.success(f"Your feedback is *{fb}*. Thank you!")
        else:
            st.sidebar.warning("Please enter some feedback.")
