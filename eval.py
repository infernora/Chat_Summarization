import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Device selection
if torch.backends.mps.is_available():  #macOS gpu acceleration
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



rouge = evaluate.load("rouge") #it compares the overlap of n-grams between the generated summary and reference summaries.
bertscore = evaluate.load("bertscore") #Measures semantic similarity using contextual embeddings


MODEL_PATH = "outputs/model/checkpoint-27750"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


df = pd.read_excel("data/medical_dialogue_test.xlsx")

preds = []
refs = []

with torch.no_grad():
    for row in df.itertuples(): #iterates through each row of the dataframe
        inputs = tokenizer(
            row.dialogue,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {k: v.to(device) for k, v in inputs.items()} #move input tensors to the same device as the model (GPU or CPU) for faster processing and to avoid device mismatch errors during generation

        outputs = model.generate(
            **inputs,
            max_length=700,
            num_beams=4,
            repetition_penalty=1.2, #penalty to discourage the model from repeating the same phrases
            early_stopping=True
        )

        summary = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        preds.append(summary)
        refs.append(row.soap)


print("\n===== ROUGE =====")
print(rouge.compute(predictions=preds, references=refs)) #higher ROUGE scores indicate better overlap

print("\n===== BERTScore =====")
print(
    bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en"
    )
)


i = 0
print("\n===== SAMPLE =====")
print("\nDIALOGUE:\n", df.iloc[i]["dialogue"])
print("\nREFERENCE:\n", df.iloc[i]["soap"])
print("\nMODEL:\n", preds[i])
