# 🌏 Machine Translation EN → VI  
**Fine-tuning a Transformer model for English → Vietnamese translation**

---

## 📘 1. Giới thiệu

Dự án này huấn luyện mô hình **dịch máy (Machine Translation)** từ tiếng **Anh → Việt**  
dựa trên kiến trúc **Transformer (MarianMT)** sử dụng thư viện **Hugging Face Transformers**.

### 🎯 Mục tiêu
- Tìm hiểu quy trình huấn luyện mô hình Seq2Seq (Sequence-to-Sequence)
- Áp dụng fine-tuning mô hình `Helsinki-NLP/opus-mt-en-vi`
- Tạo API/inference để dịch văn bản tiếng Anh sang tiếng Việt

---

## 🧠 2. Kiến thức nền tảng

### 🔹 Cấu trúc Transformer
- **Encoder**: mã hóa câu tiếng Anh thành vector ngữ nghĩa
- **Decoder**: giải mã vector ngữ nghĩa thành câu tiếng Việt
- Cơ chế **Attention** giúp mô hình “tập trung” vào các từ quan trọng trong ngữ cảnh

### 🔹 Huấn luyện Seq2Seq
- Input: Câu tiếng Anh (source)
- Target: Câu tiếng Việt (target)
- Loss function: Cross-Entropy giữa câu dự đoán và câu thật
- Optimizer: AdamW, Learning Rate Scheduler, Warmup Steps

---

## ⚙️ 3. Cấu trúc thư mục
MachineTranslation/
│
├── src/
│ ├── train.py # Script huấn luyện
│ ├── inference.py # Script suy luận
│ ├── utils.py # (tuỳ chọn) các hàm phụ trợ
│
├── data/
│ ├── train.en # Dữ liệu tiếng Anh
│ ├── train.vi # Dữ liệu tiếng Việt
│ ├── val.en
│ ├── val.vi
│
├── checkpoints/ # Lưu checkpoint sau mỗi bước huấn luyện
│ ├── checkpoint-49000/
│ ├── checkpoint-49500/
│ ├── checkpoint-49995/
│ └── ... + model cuối cùng
│
├── out/ # (tùy chọn) Model đã chọn để inference
│
├── requirements.txt # Danh sách thư viện cần cài
└── README.md # File hướng dẫn

---

## 🧩 4. Cài đặt môi trường

### Yêu cầu hệ thống
- Python ≥ 3.9  
- GPU có CUDA (khuyến nghị)
- pip hoặc conda environment

### Bước cài đặt

```bash
# Clone project
git clone https://github.com/<your-username>/MachineTranslation.git
cd MachineTranslation

# Tạo virtual environment
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

