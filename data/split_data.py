import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# --- Cấu hình đường dẫn file ---
FILE_EN = "slang/data.en"
FILE_VI = "slang/data.vi"

# --- Cấu hình tỷ lệ chia dữ liệu ---
# Ví dụ: 80% Train, 10% Validation, 10% Test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Đảm bảo tổng tỷ lệ là 1.0
if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:
    raise ValueError("Tổng tỷ lệ chia phải bằng 1.0")

# --- Bước 1: Đọc Dữ Liệu ---
try:
    # Đọc dữ liệu từ file, mỗi dòng là một câu
    with open(FILE_EN, 'r', encoding='utf-8') as f_en:
        data_en = f_en.readlines()
    with open(FILE_VI, 'r', encoding='utf-8') as f_vi:
        data_vi = f_vi.readlines()
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file {e.filename}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Kiểm tra số lượng câu
if len(data_en) != len(data_vi):
    print("Lỗi: Số lượng câu tiếng Anh và tiếng Việt không khớp.")
    exit()

# Loại bỏ các ký tự xuống dòng ('\n') ở cuối mỗi câu
data_en = [line.strip() for line in data_en]
data_vi = [line.strip() for line in data_vi]

# Tạo DataFrame để dễ dàng quản lý và trộn
df = pd.DataFrame({'en': data_en, 'vi': data_vi})

# --- Bước 2: Trộn và Chia Dữ Liệu ---

# Chia tập Test trước (ví dụ 10%)
# Tỷ lệ test được tính trên tổng số dữ liệu (TEST_RATIO)
df_train_val, df_test = train_test_split(
    df, 
    test_size=TEST_RATIO, 
    random_state=42, # Đảm bảo việc trộn (shuffle) giống nhau mỗi lần chạy
    shuffle=True
)

# Chia tập Validation từ tập Train_Val còn lại
# Tỷ lệ validation được tính trên tập df_train_val (VAL_RATIO / (TRAIN_RATIO + VAL_RATIO))
val_ratio_in_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

df_train, df_val = train_test_split(
    df_train_val, 
    test_size=val_ratio_in_train_val, 
    random_state=42,
    shuffle=True
)

# --- Bước 3: Lưu Dữ Liệu Đã Chia ---

def save_split(dataframe, name):
    """Lưu dữ liệu tiếng Anh và tiếng Việt ra file riêng biệt."""
    # Lưu tiếng Anh
    with open(f'slang/{name}.en', 'w', encoding='utf-8') as f_en:
        f_en.write('\n'.join(dataframe['en']))
    # Lưu tiếng Việt
    with open(f'slang/{name}.vi', 'w', encoding='utf-8') as f_vi:
        f_vi.write('\n'.join(dataframe['vi']))
    
    print(f"Đã lưu {len(dataframe)} cặp dữ liệu vào tập {name} ({name}.en và {name}.vi)")


save_split(df_train, 'train')
save_split(df_val, 'val')
save_split(df_test, 'test')

# --- In tóm tắt kết quả ---
print("\n--- Tóm Tắt Kết Quả ---")
print(f"Tổng số cặp câu: {len(df)}")
print(f"Tập Train: {len(df_train)} cặp (Tỷ lệ: {len(df_train)/len(df):.2f})")
print(f"Tập Validation: {len(df_val)} cặp (Tỷ lệ: {len(df_val)/len(df):.2f})")
print(f"Tập Test: {len(df_test)} cặp (Tỷ lệ: {len(df_test)/len(df):.2f})")