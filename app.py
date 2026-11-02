import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="影像處理與人臉分析工具", layout="wide")
st.title("互動式影像處理與人臉分析工具")

# ==========================================
# 1. 影像載入與基礎解析
# ==========================================
st.header("1. 影像載入與基礎解析")
uploaded_file = st.file_uploader("請上傳影像 (支援 PNG, JPG, JPEG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # 讀取影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 顯示基礎中繼資訊
    height, width, channels = original_img.shape
    file_size_kb = uploaded_file.size / 1024
    st.write(f"**影像解析度**: {width} x {height} 像素")
    st.write(f"**色彩維度**: {channels} 通道")
    st.write(f"**檔案大小**: {file_size_kb:.2f} KB")
    
    # ==========================================
    # 2. 多色彩空間轉換與同步直方圖分析
    # ==========================================
    st.header("2. 多色彩空間轉換與直方圖分析")
    view_mode = st.selectbox("選擇檢視模式", 
        ["原始影像 (Original)", "R 通道", "G 通道", "B 通道", "Y 通道 (亮度)", "U 通道 (色度)", "V 通道 (色度)"])
    
    col1, col2 = st.columns(2)
    
    display_img = original_img_rgb
    hist_channel = None
    
    # 處理色彩空間轉換
    if view_mode == "原始影像 (Original)":
        display_img = original_img_rgb
    elif view_mode == "R 通道":
        display_img = np.zeros_like(original_img_rgb)
        display_img[:,:,0] = original_img_rgb[:,:,0]
        hist_channel = original_img_rgb[:,:,0]
    elif view_mode == "G 通道":
        display_img = np.zeros_like(original_img_rgb)
        display_img[:,:,1] = original_img_rgb[:,:,1]
        hist_channel = original_img_rgb[:,:,1]
    elif view_mode == "B 通道":
        display_img = np.zeros_like(original_img_rgb)
        display_img[:,:,2] = original_img_rgb[:,:,2]
        hist_channel = original_img_rgb[:,:,2]
    elif "通道" in view_mode and ("Y" in view_mode or "U" in view_mode or "V" in view_mode):
        yuv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
        if "Y" in view_mode:
            display_img = cv2.cvtColor(yuv_img[:,:,0], cv2.COLOR_GRAY2RGB)
            hist_channel = yuv_img[:,:,0]
        elif "U" in view_mode:
            display_img = cv2.cvtColor(yuv_img[:,:,1], cv2.COLOR_GRAY2RGB)
            hist_channel = yuv_img[:,:,1]
        elif "V" in view_mode:
            display_img = cv2.cvtColor(yuv_img[:,:,2], cv2.COLOR_GRAY2RGB)
            hist_channel = yuv_img[:,:,2]

    with col1:
        st.image(display_img, caption=view_mode, use_container_width=True)
        
    with col2:
        if view_mode == "原始影像 (Original)":
            # 繪製 RGB 三通道直方圖
            colors = ('r', 'g', 'b')
            hist_dict = {}
            for i, col_name in enumerate(colors):
                hist = cv2.calcHist([original_img_rgb], [i], None, [256], [0, 256]).flatten()
                hist_dict[col_name.upper()] = hist
            st.line_chart(hist_dict)
        else:
            # 繪製單一通道直方圖
            hist = cv2.calcHist([hist_channel], [0], None, [256], [0, 256]).flatten()
            st.bar_chart(hist)

    # ==========================================
    # 3. 自訂卷積核空間濾波與量化評估
    # ==========================================
    st.header("3. 自訂卷積核與 PSNR 評估")
    kernel_input = st.text_area("請輸入自訂二維卷積核矩陣 (以逗號分隔數值，換行代表新的一列)", 
                                value="0, -1, 0\n-1, 5, -1\n0, -1, 0")
    normalize = st.checkbox("啟用矩陣正規化 (Normalization)")
    
    try:
        # 解析卷積核
        lines = kernel_input.strip().split('\n')
        kernel_matrix = np.array([[float(x) for x in line.split(',')] for line in lines])
        
        if normalize:
            k_sum = np.sum(kernel_matrix)
            if k_sum != 0:
                kernel_matrix = kernel_matrix / k_sum
                
        # 進行卷積濾波
        filtered_img_rgb = cv2.filter2D(original_img_rgb, -1, kernel_matrix)
        
        st.write("**卷積核矩陣:**")
        st.write(kernel_matrix)
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.image(original_img_rgb, caption="濾波前 (原圖)", use_container_width=True)
        with f_col2:
            st.image(filtered_img_rgb, caption="濾波後結果", use_container_width=True)
            
        # 計算 PSNR
        mse = np.mean((original_img_rgb.astype(np.float64) - filtered_img_rgb.astype(np.float64)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 10 * math.log10((max_pixel ** 2) / mse)
            
        st.metric(label="峰值信噪比 (PSNR)", value=f"{psnr:.2f} dB")
        
    except Exception as e:
        st.error("卷積核格式輸入錯誤，請確認輸入數值與格式。")

    # ==========================================
    # 4. 人臉偵測模組 (OpenCV 容錯版)
    # ==========================================
    st.header("4. 人臉偵測模組 (OpenCV)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        scale_factor = st.slider("偵測靈敏度 (Scale Factor)", 1.05, 1.50, 1.10, 0.01)
    with col_b:
        min_neighbors = st.slider("相鄰矩形門檻 (minNeighbors)", 1, 15, 5, 1)
    
    xml_path = 'haarcascade_frontalface_default.xml'
    
    # 檢查檔案是否存在
    if not os.path.exists(xml_path):
        st.error(f"找不到模型檔案 '{xml_path}'，請確認是否已正確上傳至 GitHub。")
    else:
        face_cascade = cv2.CascadeClassifier()
        # 使用 load 方法並進行例外處理
        success = face_cascade.load(cv2.samples.findFile(xml_path) if hasattr(cv2.samples, 'findFile') else xml_path)
        
        if not success or face_cascade.empty():
            # 備用載入法
            face_cascade = cv2.CascadeClassifier(xml_path)
            
        if face_cascade.empty():
            st.error("模型載入失敗！請嘗試重新上傳 XML 檔案。")
        else:
            gray_img = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
            
            annotated_image = original_img_rgb.copy()
            detected_count = len(faces)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(annotated_image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
            st.image(annotated_image, caption="人臉偵測結果", use_container_width=True)
            st.success(f"**偵測到的人臉總數**: {detected_count}")
