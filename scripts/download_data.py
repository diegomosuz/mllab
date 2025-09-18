import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
df.to_csv("data/titanic.csv", index=False)
print("Dataset guardado en data/titanic.csv")
