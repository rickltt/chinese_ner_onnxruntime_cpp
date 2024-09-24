import os
import random

def split_data(data_dir, split_ratio=0.9):
    data = open(os.path.join(data_dir,"train.txt"),
                     "r", encoding="utf-8").readlines()

    test_dataset = open(os.path.join(data_dir,"test_label.txt"),
                     "r", encoding="utf-8").readlines()
        
    random.shuffle(data)
    n_total = len(data)

    train_offset = int(n_total * split_ratio)
    train_dataset = data[:train_offset]
    val_dataset = data[train_offset:]

    train_file = open(os.path.join(data_dir,'processed_train.txt'), 'w', encoding="utf-8")
    val_file = open(os.path.join(data_dir,'processed_val.txt'), 'w', encoding="utf-8")
    test_file = open(os.path.join(data_dir,'processed_test.txt'), 'w', encoding="utf-8")

    for item in train_dataset:
        train_file.write(item)

    for item in val_dataset:
        val_file.write(item)

    for item in test_dataset:
        test_file.write(item)

    train_file.close()
    val_file.close()
    test_file.close()

if __name__ == '__main__':
    split_data("./dataset")