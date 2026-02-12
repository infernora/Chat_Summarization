import pandas as pd


df = pd.read_csv("data/medical_dialogue_train.csv")
print(df.columns)
print(df.iloc[0])
