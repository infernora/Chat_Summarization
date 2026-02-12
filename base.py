import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#baseline model for comparison
MODEL_NAME = "facebook/bart-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Device selection
if torch.backends.mps.is_available():  #macOS gpu acceleration
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)

df = pd.read_excel("data/medical_dialogue_test.xlsx")

def summarize(text):
    inputs = tokenizer(      #converts text to token numbers
        text,
        return_tensors="pt",  #returns PyTorch tensors
        truncation=True,      #truncates text to fit model's max input length
        max_length=1024
    ).to(device)

    with torch.no_grad():
        output = model.generate( #produces summary tokens
            **inputs,
            max_length=700, 
            num_beams=4,    #beam search width, keeps top 4 sequences at each step
            length_penalty=2.0,  #penalty that encourages shorter summaries, higher value means longer summaries
        )

    return tokenizer.decode(output[0], skip_special_tokens=True) #converts token numbers back to text, skipping special tokens like <s>, </s>, <pad>, etc.

print("\nBASELINE SUMMARY:\n")
print(summarize(df.iloc[0]["dialogue"]))
