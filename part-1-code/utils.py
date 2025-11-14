import datasets
from datasets import load_dataset
from datasets.iterable_dataset import Key
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example

def _get_synonym(word):
  # find synonym inn wordnet or None
  synsets = wordnet.synsets(word)
  if not synsets:
    return None
  # get all lemma from first synset as candidates
  lemmas = [lemma.name().replace("_", " ") for lemma in synsets[0].lemmas()]
  candidates = [l for l in lemmas if l.lower() != word.lower()]
  if not candidates:
    return None
  return random.choice(candidates)

KEYBOARD_NEIGHBORS = {
    "a": "qswzx",
    "e": "wsdrf",
    "i": "ujklo",
    "o": "iklp",
    "u": "yhjik",
}

def _add_typo(word):
  indices = [i for i, ch in enumerate(word) if ch.lower() in KEYBOARD_NEIGHBORS]
  if not indices:
    return word

  idx = random.choice(indices)
  ch = word[idx]
  neighbors = KEYBOARD_NEIGHBORS[ch.lower()]
  new_ch = random.choice(neighbors)
  if ch.isupper():
    new_ch = new_ch.upper()

  return word[:idx] + new_ch + word[idx + 1:]

### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]

    tokens = word_tokenize(text)
    new_tokens = []

    for tok in tokens:
      if tok.isalpha():
        r = random.random()
        if r < 0.4:
          synonym = _get_synonym(tok)
        elif r < 0.7: 
          tok = _add_typo(tok)
      new_tokens.append(tok)

    # detokenize to sentence
    detok = TreebankWordDetokenizer().detokenize(new_tokens)
    example['text'] = detok

    ##### YOUR CODE ENDS HERE ######

    return example
