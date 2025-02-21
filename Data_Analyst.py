import streamlit as st
import pandas as pd
import numpy as np
import io
import csv
from io import BytesIO
import zipfile
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz
import math

st.set_page_config(page_title="Optimize Affiliate Data", layout="wide")
st.title("📊 Optimize Affiliate Data")
def read_csv_safe(uploaded_file):
    try:
        return pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8-sig')),
                           quoting=csv.QUOTE_MINIMAL, skipinitialspace=True,
                           engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Lỗi đọc CSV: {e}")
    return pd.DataFrame()

def validate_columns(df):
    required_columns = ['Price', 'Link']
    if not set(required_columns).issubset(df.columns):
        st.error(f"Thiếu cột bắt buộc: {', '.join(set(required_columns) - set(df.columns))}")
        return pd.DataFrame()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df[df['Link'].astype(str).str.match(r'^(http|https)://', na=False)]
    return df.dropna(subset=required_columns)

def preprocess_sold_column(df):
    if 'Sold' in df.columns:
        df['Sold'] = pd.to_numeric(
            df['Sold'].astype(str)
            .str.replace(r'(/tháng|lượt bán|,)', '', regex=True)
            .str.replace('k', '000', regex=False),
            errors='coerce')
    return df

def process_group(group_df, input_ratio):
    # Gộp tất cả giá trị Sold thành chuỗi, cách nhau dấu phẩy
    grouped = group_df.groupby("Link").agg({
        "Name": "first",  
        "Sold": lambda x: ",".join(map(str, x))  
    }).reset_index()
    
    # Hàm tính độ tăng trưởng
    def calculate_growth_rate(sold_str):
        sold_list = list(map(int, sold_str.split(',')))  
        if len(sold_list) < 2:  
            return -999  # Nếu chỉ có 1 số, trả về -999
        if len(sold_list) == 2:
            return 100  # Nếu chỉ có 2 số, mặc định trả về 100
    
    # Tính tốc độ tăng trưởng theo công thức tùy chỉnh
        growth_rates = [((sold_list[i+1] - sold_list[i]) / (sold_list[i] - sold_list[i-1])) * 100 
                    for i in range(1, len(sold_list) - 1)]
    
        return np.mean(growth_rates)
 
    # Tính Growth Rate
    grouped["Growth Rate"] = grouped["Sold"].apply(calculate_growth_rate)
    grouped["Sold"] = grouped["Sold"].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    # Chuyển cột Sold thành số nguyên nếu chỉ có 1 giá trị
    def process_sold(sold_str):
        sold_values = list(map(float, sold_str.split(',')))  # Chuyển đổi thành danh sách số nguyên

        if len(sold_values) == 1:
            return sold_values[0]  # Nếu chỉ có 1 giá trị, giữ nguyên
        else:
            return sold_values

    # Lọc trùng lặp dựa vào tỷ lệ giống nhau của tên sản phẩm
    filtered_df = pd.DataFrame(columns=grouped.columns)  
    for _, row in grouped.iterrows():
        if not any(fuzz.ratio(row["Name"], other_name) >= input_ratio * 100 for other_name in filtered_df["Name"]):
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
# Sắp xếp dữ liệu
    above_1 = filtered_df[filtered_df["Growth Rate"] > -999].sort_values(by="Growth Rate", ascending=False)
# Với Growth Rate == -999, đảm bảo "Sold" được so sánh dưới dạng số
    filtered_df["Sold"] = filtered_df["Sold"].apply(pd.to_numeric, errors="coerce")
    equal_1 = filtered_df[filtered_df["Growth Rate"] == -999].copy()
    equal_1 = equal_1.sort_values(by="Sold", ascending=False)

# Gộp lại thành kết quả cuối cùng

    return pd.concat([above_1, equal_1], ignore_index=True)

def remove_similar_keywords(df, input_ratio):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_group, df[df['Keyword'] == keyword], input_ratio) for keyword in df['Keyword'].unique()]
        for future in futures:
            result = future.result()
            if not result.empty:
                results.append(result)  # Dùng append thay vì extend để tránh lỗi

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
def split_and_download_csv(above_df, equal_df, max_rows=500):
    total_rows = len(above_df) + len(equal_df)
    num_files = math.ceil(total_rows / max_rows)
    files = []
    
    # Chuyển toàn bộ dữ liệu của cột cần xử lý về chuỗi để tránh lỗi
    for col in above_df.columns:
        if col in equal_df.columns:
            above_df[col] = above_df[col].astype(str)
            equal_df[col] = equal_df[col].astype(str)
    
    # Chia đều dữ liệu vào các file
    above_chunks = [above_df[i::num_files] for i in range(num_files)]
    equal_chunks = [equal_df[i::num_files] for i in range(num_files)]
    
    for i in range(num_files):
        # Ghép dữ liệu từ 2 nhóm
        combined_chunk = pd.concat([above_chunks[i], equal_chunks[i]], ignore_index=True)
        
        # Lưu vào file CSV ảo
        csv_buffer = BytesIO()
        combined_chunk.to_csv(csv_buffer, index=False, encoding='utf-8-sig', quoting=1)  # quoting=1 để tránh lỗi chuyển đổi
        csv_buffer.seek(0)
        
        files.append((f"filtered_data_part_{i+1}.csv", csv_buffer))
    
    return files

def create_zip(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, file_buffer in files:
            zip_file.writestr(filename, file_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer


uploaded_files = st.file_uploader("Chọn nhiều file CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    combined_df = pd.concat([
        validate_columns(preprocess_sold_column(read_csv_safe(file))) for file in uploaded_files
    ], ignore_index=True)

    if not combined_df.empty:
        st.subheader("📌 Dữ liệu hợp lệ ")
        st.dataframe(combined_df.sort_index(ascending=False), height=500,width=1500)

        number = st.number_input(label="Tỉ lệ trùng lặp:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        if st.button("🔍 Lọc dữ liệu ") and number != 0:
            combined_df.dropna(subset=['Sold'], inplace=True)
            combined_df = remove_similar_keywords(combined_df, number)
            st.session_state.df_combined = combined_df
            above_df=st.session_state.df_combined[st.session_state.df_combined["Growth Rate"] > -999].sort_values(by="Growth Rate",ascending=False)
            equal_df=st.session_state.df_combined[st.session_state.df_combined["Growth Rate"] == -999].sort_values(by="Sold",ascending=False)
            st.subheader("📌 Dữ liệu được cập nhật nhiều ngày")
            st.dataframe(above_df, height=500, width=1500)
            st.subheader("�� Dữ liệu chỉ chưa cập nhật")
            st.dataframe(equal_df, height=500, width=1500)

            files = split_and_download_csv(above_df, equal_df)
            zip_buffer = create_zip(files)
    
            st.subheader("📥 Tải xuống toàn bộ CSV dưới dạng ZIP")
            st.download_button(
            label="📁 Tải xuống tất cả (ZIP)",
            data=zip_buffer,
            file_name="filtered_data.zip",
            mime="application/zip"
    )
    else:
        st.warning("Không có dữ liệu hợp lệ sau khi kiểm tra.")
else:
    st.warning("Vui lòng tải lên ít nhất một file CSV.")
