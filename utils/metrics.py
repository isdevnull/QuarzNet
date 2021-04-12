import editdistance
from typing import List

def WER(hypothesis:List[str], reference:List[str]) -> int:
  return float(editdistance.eval(hypothesis, reference)) / len(reference)

def CER(hypothesis: str, reference: str) -> float:
  return float(editdistance.eval(hypothesis, reference)) / len(reference)