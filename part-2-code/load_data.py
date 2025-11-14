import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# ============== 在这里放全局 Tokenizer（最正确的位置） ==============
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
BOS_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<extra_id_0>")
PAD_IDX = TOKENIZER.pad_token_id
# ================================================================


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.split = split
        self.data_folder = data_folder
        self.max_length = 512
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.data = self.process_data(data_folder, split)


    def process_data(self, data_folder, split):
      # TODO
      # NL
      nl_path = os.path.join(data_folder, f"{split}.nl")
      with open(nl_path, "r") as f:
        nl_queries = [line.strip() for line in f.readlines()]

      processed = []

      if split == "test":
        # test 没有 SQL， encoder 's tokenize
        for nl in nl_queries:
          nl_prompt = "Convert this natural language question into SQL: " + nl
          enc = self.tokenizer(
            nl_prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
          )
          processed.append(
            {
              "encoder_input_ids": enc["input_ids"].squeeze(0),
              "encoder_attention_mask": enc["attention_mask"].squeeze(0),
              "decoder_input_ids": None,
              "decoder_targets": None,
            }
          )
      else:
        # train / dev：有 SQL
        sql_path = os.path.join(data_folder, f"{split}.sql")
        with open(sql_path, "r") as f:
          sql_queries = [line.strip() for line in f.readlines()]

        assert len(nl_queries) == len(
          sql_queries
        ), f"Mismatch NL/SQL: {len(nl_queries)} vs {len(sql_queries)}"

        for nl, sql in zip(nl_queries, sql_queries):
          # encoder：带一个简单任务前缀
          nl_prompt = "Convert this natural language question into SQL: " + nl
          enc = self.tokenizer(
            nl_prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
          )

          # decoder：SQL 序列
          dec = self.tokenizer(
            sql,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
          )

          processed.append(
            {
              "encoder_input_ids": enc["input_ids"].squeeze(0),
              "encoder_attention_mask": enc["attention_mask"].squeeze(0),
              "decoder_input_ids": dec["input_ids"].squeeze(0),
              "decoder_targets": dec["input_ids"].squeeze(0),
            }
          )
      return processed
    
    def __len__(self):
        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # 
    encoder_input_ids = [item["encoder_input_ids"] for item in batch]
    encoder_attention_mask = [item["encoder_attention_mask"] for item in batch]
    decoder_input_ids_list = [item["decoder_input_ids"] for item in batch]

    # pad encoder
    encoder_ids = pad_sequence(
        encoder_input_ids, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = pad_sequence(
        encoder_attention_mask, batch_first=True, padding_value=0
    )

    bos_token_id = BOS_TOKEN_ID
    decoder_inputs_list = []
    decoder_targets_list = []

    for dec_ids in decoder_input_ids_list:
        # decoder input：BOS + y[:-1]（右移一位）
        decoder_input = torch.cat(
            [torch.tensor([bos_token_id], dtype=torch.long), dec_ids[:-1]]
        )
        decoder_inputs_list.append(decoder_input)

        # target：原始序列
        decoder_targets_list.append(dec_ids)

    # pad decoder
    decoder_inputs = pad_sequence(
        decoder_inputs_list, batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        decoder_targets_list, batch_first=True, padding_value=PAD_IDX
    )

    # initial decoder inputs：每个样本只要一个 BOS
    initial_decoder_inputs = torch.full(
        (len(batch), 1), bos_token_id, dtype=torch.long
    )

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_input_ids = [item["encoder_input_ids"] for item in batch]
    encoder_attention_mask = [item["encoder_attention_mask"] for item in batch]

    encoder_ids = pad_sequence(
        encoder_input_ids, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = pad_sequence(
        encoder_attention_mask, batch_first=True, padding_value=0
    )

    bos_token_id = BOS_TOKEN_ID

    initial_decoder_inputs = torch.full(
        (len(batch), 1), bos_token_id, dtype=torch.long
    )

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x   = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y   = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x  = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x