import os
from tqdm import tqdm
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, args, split):
        fname = os.path.join(args.data_dir, 'processed_{}.txt'.format(split))
        data = open(fname, 'r', encoding='utf-8').readlines()

        self.samples = []
        data_iterator = tqdm(data, desc="Loading: {} Data".format(split))

        self.max_seq_length = args.max_seq_length
        self.tokenizer = args.tokenizer

        self.label2id = args.label2id
        self.id2label = args.id2label

        pad_token_label_id = -100
        for instance in data_iterator:
            tokens, labels = [], []
            for item in instance.rstrip().split(" "):
                word, tag = item.split("/")
                word_tokens = self.tokenizer.tokenize(word)
                if word_tokens == []:
                    word_tokens = [self.tokenizer.unk_token]
                for word_token in word_tokens:
                    tokens.append(word_token)
                if tag != 'o':
                    labels.append(f"B-{tag.upper()}")
                    for _ in range(len(word_tokens)-1):
                        labels.append(f"I-{tag.upper()}")
                else:
                    labels.extend(['O'] * len(word_tokens))
            assert len(tokens) == len(labels)

            label_ids = [ self.label2id[i] for i in labels]

            special_tokens_count = 2
            if len(tokens) > self.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
                # valid_mask = valid_mask[: (self.max_seq_length - special_tokens_count)]
            
            # add sep token
            tokens += ['[SEP]']
            label_ids += [pad_token_label_id]
            # valid_mask.append(1)
            segment_ids = [0] * len(tokens)

            # add cls token
            tokens = ['[CLS]'] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [0] + segment_ids
            # valid_mask.insert(0, 1)

            # input_ids, attention_mask
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # padding
            padding_length = self.max_seq_length - len(input_ids)
            input_ids += [0] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [0] * padding_length
            # valid_mask += [0] * padding_length
            while (len(label_ids) < self.max_seq_length):
                label_ids.append(pad_token_label_id)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length
            # assert len(valid_mask) == self.max_seq_length

            sample = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                # 'valid_mask': valid_mask,
                'label_ids': label_ids
            }
            self.samples.append(sample)

    def __getitem__(self, index):
        return self.samples[index] 
    def __len__(self):
        return len(self.samples)  
