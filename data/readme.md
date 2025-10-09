Chúng ta cần dữ liệu có dạng:
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

mỗi phần tử có dạng:
{
  "translation": {
    "en": "I love studying AI.",
    "vi": "Tôi thích học trí tuệ nhân tạo."
  }
}