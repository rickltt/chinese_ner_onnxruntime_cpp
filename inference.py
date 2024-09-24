from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


model_name_or_path = "output/best_checkpoint"
# label_list = ["O", "B-NR", "I-NR", "B-NS", "I-NS", "B-NT", "I-NT"]
# num_labels = len(label_list)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=256)
# model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
pipe = pipeline("token-classification", model=model_name_or_path, device=0)

if __name__ == '__main__':
    text = "中共中央致中国致公党十一大的贺词"
    prediction = pipe(text)
    print(prediction)