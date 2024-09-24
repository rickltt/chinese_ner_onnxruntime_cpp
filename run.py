import torch
import numpy as np
import random
import argparse
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm, trange
from data_utils import NERDataset
from utils import collate_fn, set_seed
from model import BertForNER, BertForNERCRF
from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import logging
logger = logging.getLogger(__name__)

def train(args, model, train_dataset, dev_dataset):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size 
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps

    print('logging_steps:',args.logging_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_dev_f1 = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            model.train()

            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                # "valid_mask":batch[3],
                "label_ids":batch[3],
                "mode": "train"
            }

            loss = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        result, _ = evaluate(args, model, dev_dataset)
                        if best_dev_f1 < result['f1']:
                            best_dev_f1 = result['f1']
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model.save_pretrained(output_dir)
                            # torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                            args.tokenizer.save_pretrained(output_dir)

                            # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            # logger.info("Saving model checkpoint to %s", output_dir)

                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            # logger.info("Saving optimizer and scheduler states to %s", output_dir)
        if args.evaluate_after_epoch:
            result, _ = evaluate(args, model, dev_dataset)
            output_dir = os.path.join(args.output_dir, "best_checkpoint")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            # torch.save(model.state_dict(), os.path.join(output_dir, "model"))
            args.tokenizer.save_pretrained(output_dir)
            # torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset):

    args.eval_batch_size = args.per_gpu_eval_batch_size 
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    trues = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch.values())
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                # "valid_mask":batch[3],
                "label_ids":batch[3],
                "mode": "test"
            }
            logits, tmp_eval_loss = model(**inputs)
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            trues = inputs["label_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            trues = np.append(trues, inputs["label_ids"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    print('eval_loss={}'.format(eval_loss))
    if not args.use_crf:
        preds = np.argmax(preds, axis=2)

    trues_list = [[] for _ in range(trues.shape[0])]
    preds_list = [[] for _ in range(preds.shape[0])]

    pad_token_label_id = - 100
    for i in range(trues.shape[0]):
        for j in range(trues.shape[1]):
            if trues[i, j] != pad_token_label_id:
                trues_list[i].append(args.id2label[trues[i][j]])
                preds_list[i].append(args.id2label[preds[i][j]])
    precision = precision_score(trues_list, preds_list)
    recall = recall_score(trues_list, preds_list)
    f1 = f1_score(trues_list, preds_list)
    overall_result = classification_report(trues_list, preds_list)
    print(overall_result)

    metric = 'eval_loss={}\n'.format(eval_loss)
    metric += '[MSRA Result]\tP={:.4f}\tR={:.4f}\tF1={:.4f}\n'.format(precision, recall, f1)

    print(metric)
    
    result = {}
    result['eval_loss'] = eval_loss
    result['p'] = precision
    result['r'] = recall
    result['f1'] = f1

    return result, metric


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--data_dir', default='dataset', type=str)
    parser.add_argument("--model_name_or_path", default='hfl/chinese-roberta-wwm-ext', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--use_crf", action="store_true",
                        help="Whether to use crf.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--evaluate_after_epoch", action="store_true",
                        help="Whether to run evaluation after each epoch.")
    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="dropout_prob.")
    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=str, default='0.5',
                        help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer

    label_list = ["O", "B-NR", "I-NR", "B-NS", "I-NS", "B-NT", "I-NT"]
    label2id = {j:i for i,j in enumerate(label_list)}
    id2label = {i:j for i,j in enumerate(label_list)}
    num_labels = len(label_list)

    args.label2id = label2id
    args.id2label = id2label
    args.num_labels = num_labels

    datasets = {}
    data_splits = ["train","val","test"]
    for split in data_splits:
        datasets[split] = NERDataset(args, split)

    # Set seed
    set_seed(args)

    if args.use_crf:
        model = BertForNERCRF.from_pretrained(
            args.model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    else:
        model = BertForNER.from_pretrained(
            args.model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
    for param in model.parameters(): param.data = param.data.contiguous()

    model.to(args.device)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, datasets["train"], datasets["val"])
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # torch.save(args, os.path.join(output_dir, "training_args.bin"))
        
        # logger.info("Saving model checkpoint to %s", output_dir)

    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'last_checkpoint')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = BertForNER.from_pretrained(checkpoint)
        # state_dict = torch.load(os.path.join(checkpoint, "model"))
        # model.load_state_dict(state_dict)
        model.to(args.device)

        _, metric = evaluate(args, model, datasets["test"])
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            # writer.write('***** Model: {} Predict in {} test dataset *****'.format(args.model_type, args.dataset))
            writer.write("{} \n".format(metric))
        
if __name__ == '__main__':
    main()