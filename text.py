from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("outputs/model", use_fast=False)
print(type(tok))
