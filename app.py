import streamlit as st
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from PIL import Image
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError as e:
    st.error(f"Error importing required libraries: {e}")
    st.stop()

import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Deteksi Abnormalitas Sel Sperma",
    page_icon="🔬",
    layout="wide"
)

# Load model dengan error handling
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

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
    
    # Opsi upload gambar
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
            try:
                results = model.predict(
                    image, 
                    imgsz=640,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
            except Exception as e:
                st.error(f"Error selama prediksi: {e}")
                st.stop()
        
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
    st.info("Fitur evaluasi memerlukan ground truth labels untuk bekerja dengan baik. Fitur ini dalam pengembangan.")

with tab3:
    st.header("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk mendeteksi abnormalitas pada bentuk kepala sel sperma menggunakan model YOLOv8.
    
    ### Cara Penggunaan:
    1. Pada tab **Deteksi Gambar**, upload gambar sperma
    2. Atur threshold confidence dan IOU sesuai kebutuhan di sidebar
    3. Lihat hasil deteksi dan statistiknya
    
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