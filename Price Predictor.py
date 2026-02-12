import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error

# Datasets

filepath = [
    "/content/drive/MyDrive/Data.csv",
    "/content/drive/MyDrive/2.csv",
    "/content/drive/MyDrive/3.csv",
    "/content/drive/MyDrive/4.csv",
    "/content/drive/MyDrive/5.csv",
    "/content/drive/MyDrive/6.csv",
    "/content/drive/MyDrive/7.csv",
    "/content/drive/MyDrive/8.csv",
    "/content/drive/MyDrive/9.csv",
    "/content/drive/MyDrive/10.csv"
]

dfs = []
for file in filepath:
    temp = pd.read_csv(file)
    dfs.append(temp)

# Combine all datasets
df = pd.concat(dfs, ignore_index=True)

print("All CSV files merged successfully!")
print("Total Rows:", len(df))

# Preprocessing

df.columns = df.columns.str.lower()

# Convert date properly
df["arrival_date"] = pd.to_datetime(df["arrival_date"], dayfirst=True)

# Target price
df["price"] = df["modal_price"]

# Sort time-series order
df = df.sort_values(["commodity", "district", "market", "arrival_date"])

# FE

def features(group):
    group = group.copy()

    for lag in [1, 7, 14, 30]:
        group[f"lag_{lag}"] = group["price"].shift(lag)

    group["rolling_7"] = group["price"].shift(1).rolling(7).mean()
    group["rolling_30"] = group["price"].shift(1).rolling(30).mean()

    return group

df = df.groupby(
    ["commodity", "district", "market"],
    group_keys=False
).apply(create_lag_features)

df = df.dropna()
df = df.reset_index(drop=True)

com = LabelEncoder()
mar = LabelEncoder()
dist = LabelEncoder()

df["commodity_id"] = com.fit_transform(df["commodity"])
df["market_id"] = mar.fit_transform(df["market"])
df["district_id"] = dist.fit_transform(df["district"])

# Model

features = [
    "commodity_id", "market_id", "district_id",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_7", "rolling_30"
]

X = df[features]
y = df["price"]

model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05
)

model.fit(X, y)

print("Model Trained Successfull")

# Prediction Part

def fam(days):

    fp = []

    latest = df.groupby(["commodity", "district", "market"]).tail(1)

    for _, row in latest.iterrows():

        inpdat = pd.DataFrame([{
            "commodity_id": row["commodity_id"],
            "market_id": row["market_id"],
            "district_id": row["district_id"],

            "lag_1": row["lag_1"],
            "lag_7": row["lag_7"],
            "lag_14": row["lag_14"],
            "lag_30": row["lag_30"],

            "rolling_7": row["rolling_7"],
            "rolling_30": row["rolling_30"]
        }])

        prep = model.predict(inpdat)[0]

        fp.append({
            "commodity": row["commodity"],
            "district": row["district"],
            "market": row["market"],
            f"forecast_price_{days}d": round(prep, 2)
        })

    return pd.DataFrame(fp)

# 6&8 Dataset

f6weeks = fam(42)
f8weeks = fam(56)

ffore = f6weeks.merge(f8weeks,on=["commodity", "district", "market"])

print("\nCompleted!")
print(final_forecast.head())

# Downloading

f6weeks.to_csv("f6weeks.csv", index=False)
f8weeks.to_csv("f8weeks.csv", index=False)
ffore.to_csv("forecast of 6&8 weeks.csv", index=False)

print("\nSaved Files:")

# Accuracy

print("\nAccuracy")

test_size = 56

train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

X_train, y_train = train[features], train["price"]
X_test, y_test = test[features], test["price"]

model_test = LGBMRegressor(n_estimators=600)
model_test.fit(X_train, y_train)

preds = model_test.predict(X_test)

mape = mean_absolute_percentage_error(y_test, preds)
accuracy = (1 - mape) * 100

print(f"\nAccuracy: {accuracy:.2f}%")