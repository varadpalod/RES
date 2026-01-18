import pandas as pd
from pathlib import Path

df = pd.read_csv('data/pune_house_prices.csv')
print("Locality Counts:")
print(df['area'].value_counts())

print("\nChecking for specific 'Koregaon' matches:")
print(df[df['area'].astype(str).str.contains('Koregaon', case=False)]['area'].value_counts())

print("\nChecking for specific 'Hinjewadi' matches:")
print(df[df['area'].astype(str).str.contains('Hinjewadi', case=False)]['area'].value_counts())


print("\nLast 20 rows:")
print(df[['id', 'area', 'price']].tail(20))

print("\nUnknown counts:")
print(df[df['area'] == 'Unknown']['area'].count())
