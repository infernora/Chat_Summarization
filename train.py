from datasets import Dataset
import pandas as pd

#data loading
train_df = pd.read_csv("data/medical_dialogue_train.csv")
val_df = pd.read_excel("data/medical_dialogue_validation.xlsx")

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

#tokenizer setup
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

#preprocessing function for tokenization and padding
def preprocess(batch):
    inputs = tokenizer(
        batch["dialogue"],  #access dialogue column from batch
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    outputs = tokenizer(
        batch["soap"],  #access soap column from batch
        max_length=700,  #adjusted based on data analysis
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = outputs["input_ids"] #add labels for seq2seq training, inputs will be fed to the model and labels will be used for loss calculation
    return inputs

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

#model setup
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

from transformers import GenerationConfig

gen_config = GenerationConfig.from_model_config(model.config) #create generation config from model's config to ensure compatibility
gen_config.forced_bos_token_id = model.config.bos_token_id #bos = beginning of sequence token, ensures generated summaries start with this token
model.generation_config = gen_config

# Data collator handles padding dynamically during training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) #Dynamically pads sequences in each batch instead of padding everything to max_length, saves memory

# Update your arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="outputs/model",
    eval_strategy="epoch",       #evaluate at the end of each epoch
    learning_rate=2e-5,      
    per_device_train_batch_size=1,   #process one dialogue-summary pair at a time
    per_device_eval_batch_size=1,
    weight_decay=0.01,              #regularization to prevent overfitting
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,  # Required for Seq2Seq metrics
    fp16=False,                  # Standard on Mac (use True only for NVIDIA)
    logging_steps=100,
    report_to="none",
    gradient_checkpointing=True  

)

# Initialize the correct Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    # tokenizer=tokenizer,         # This is now valid for Seq2SeqTrainer
    data_collator=data_collator  
)

trainer.train()
