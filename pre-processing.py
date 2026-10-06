import pandas as pd
import numpy as np

df=pd.read_csv("Most Streamed Artists on Spotify (17_07_2026) V1.1-selected-columns.csv")

# 2. Understand dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# 3. Check missing values
print(df.isnull().sum())

# 4. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Encode categorical data
df = pd.get_dummies(df, columns=["Gender"], dtype=int)

# 7. Separate X and y
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# 8. Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing completed!")