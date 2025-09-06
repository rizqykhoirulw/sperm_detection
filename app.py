import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Deteksi Abnormalitas Sel Sperma",
    page_icon="🔬",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()

# Judul aplikasi
st.title("🔬 Deteksi Abnormalitas Bentuk Kepala Sel Sperma")
st.markdown("Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi abnormalitas pada bentuk kepala sel sperma")

# Sidebar
st.sidebar.header("Pengaturan")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.01)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.65, 0.01)

# Tab untuk berbagai fungsi
tab1, tab2, tab3 = st.tabs(["Deteksi Gambar", "Evaluasi Model", "Tentang"])

with tab1:
    st.header("Deteksi pada Gambar Baru")
    
    # Opsi upload gambar atau pilih sample
    option = st.radio("Pilih sumber gambar:", ("Upload Gambar", "Gunakan Sample"))
    
    if option == "Upload Gambar":
        uploaded_file = st.file_uploader("Pilih gambar sel sperma...", type=["jpg", "jpeg", "png"])
    else:
        sample_files = os.listdir("sample_images") if os.path.exists("sample_images") else []
        if sample_files:
            selected_sample = st.selectbox("Pilih sample gambar:", sample_files)
            uploaded_file = open(os.path.join("sample_images", selected_sample), "rb")
        else:
            st.warning("Folder sample_images tidak ditemukan. Silakan upload gambar.")
            uploaded_file = st.file_uploader("Pilih gambar sel sperma...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Gambar Asli", use_column_width=True)
        
        # Run prediction
        with st.spinner("Sedang melakukan deteksi..."):
            results = model.predict(
                image, 
                imgsz=640,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
        
        # Draw bounding boxes
        result_image = image.copy()
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                
                # Draw bounding box
                color = (0, 255, 0) if label == "normal" else (0, 0, 255)
                cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(result_image, f"{label} {conf:.2f}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    "Label": label,
                    "Confidence": conf,
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2
                })
        
        # Display result image
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        with col2:
            st.image(result_image_rgb, caption="Hasil Deteksi", use_column_width=True)
        
        # Show detection results
        if detections:
            st.subheader("Hasil Deteksi")
            df = pd.DataFrame(detections)
            st.dataframe(df)
            
            # Count by class
            st.subheader("Statistik Deteksi")
            col1, col2, col3 = st.columns(3)
            normal_count = len(df[df["Label"] == "normal"])
            abnormal_count = len(df[df["Label"] == "anormal"])
            
            col1.metric("Normal", normal_count)
            col2.metric("Abnormal", abnormal_count)
            col3.metric("Total", len(df))
        else:
            st.warning("Tidak terdeteksi sel sperma pada gambar ini.")

with tab2:
    st.header("Evaluasi Model")
    
    # Upload multiple images for evaluation
    st.subheader("Evaluasi pada Multiple Gambar")
    eval_files = st.file_uploader("Pilih gambar untuk evaluasi (multiple)", 
                                 type=["jpg", "jpeg", "png"], 
                                 accept_multiple_files=True)
    
    if eval_files and st.button("Jalankan Evaluasi"):
        all_detections = []
        gt_data = []  # Anda perlu memiliki ground truth untuk evaluasi lengkap
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(eval_files):
            status_text.text(f"Memproses gambar {i+1}/{len(eval_files)}")
            
            # Process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Run prediction
            results = model.predict(
                image, 
                imgsz=640,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Collect detections
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.names[cls_id]
                    conf = box.conf[0].item()
                    
                    all_detections.append({
                        "Image": uploaded_file.name,
                        "Label": label,
                        "Confidence": conf
                    })
            
            progress_bar.progress((i + 1) / len(eval_files))
        
        status_text.text("Evaluasi selesai!")
        
        if all_detections:
            # Create summary
            df = pd.DataFrame(all_detections)
            summary = df.groupby("Label").agg(
                Count=("Label", "count"),
                Avg_Confidence=("Confidence", "mean")
            ).reset_index()
            
            st.subheader("Ringkasan Evaluasi")
            st.dataframe(summary)
            
            # Display chart
            fig, ax = plt.subplots()
            summary.set_index("Label")["Count"].plot(kind="bar", ax=ax)
            ax.set_ylabel("Jumlah Deteksi")
            ax.set_title("Distribusi Deteksi")
            st.pyplot(fig)
        else:
            st.warning("Tidak ada deteksi pada gambar-gambar yang diupload.")

with tab3:
    st.header("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk mendeteksi abnormalitas pada bentuk kepala sel sperma menggunakan model YOLOv8.
    
    ### Cara Penggunaan:
    1. Pada tab **Deteksi Gambar**, upload gambar atau pilih sample yang tersedia
    2. Atur threshold confidence dan IOU sesuai kebutuhan di sidebar
    3. Lihat hasil deteksi dan statistiknya
    4. Gunakan tab **Evaluasi Model** untuk menguji model pada multiple gambar
    
    ### Informasi Model:
    - Architecture: YOLOv8
    - Classes: Normal, Abnormal
    - Input size: 640x640 pixels
    """)
    
    # Display model information
    st.subheader("Informasi Model")
    st.text(f"Model: YOLOv8")
    st.text(f"Jumlah kelas: {len(model.names)}")
    st.text(f"Nama kelas: {list(model.names.values())}")

# Footer
st.markdown("---")
st.markdown("Dibuat dengan Streamlit dan YOLOv8 | © 2024")