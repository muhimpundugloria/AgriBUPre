import pandas as pd

df = pd.read_csv("data/agriculture_burundi.csv")

print("Dataset chargé avec succès")
print()

print("Dimensions :", df.shape)
print()

print(df.head())