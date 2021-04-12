import string
from typing import List
import torch

class TextTransform:
  def __init__(self, alphabet="'" + ' ' + string.ascii_lowercase):
    self.alphabet = alphabet
    self.int2char = dict(enumerate(alphabet))
    self.char2int = {val: key for (key, val) in self.int2char.items()}

  def text_to_int(self, text: str):
    text = text.lower()
    encoded_sequence = list(map(lambda symbol: self.char2int[symbol], text))
    return encoded_sequence

  def int_to_text(self, tokens: List[int]):
    decoded_sequence = list(map(lambda token: self.int2char[token], tokens))
    return ''.join(decoded_sequence)

def greedy_path_search(text_transform: TextTransform, hypothesis: torch.Tensor, reference: torch.Tensor, seq_length: int, blank: int=28):
  hypothesis, reference = hypothesis[:seq_length], reference[:seq_length]
  hypothesis = torch.unique_consecutive(hypothesis)
  hypothesis = hypothesis[hypothesis != blank]
  return text_transform.int_to_text(hypothesis.tolist()), text_transform.int_to_text(reference.tolist())
  