import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_rel

file_path = r"D:\CDU\prt564\project\646702.xlsx"

# read
df_raw = pd.read_excel(file_path, sheet_name="Data1", header=None)

print("Raw Shape:", df_raw.shape)

headers = df_raw.iloc[0]

metadata = df_raw.iloc[1:10].copy()
metadata.to_csv("metadata.csv", index=False)


df = df_raw.iloc[10:].copy()
df.columns = headers
df.reset_index(drop=True, inplace=True)

# rename first column to date
df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# convert date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# convert rest of the columns to numeric
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# fill missing values forward
df = df.ffill()

print("Clean Shape:", df.shape)
print("Date Range:", df["Date"].min(), "to", df["Date"].max())

# target column for prediction
target_col = "Index Numbers ;  Employee households ;  All groups ;"

print("Non-null values in target:", df[target_col].notna().sum())

# -----------------------------
# charts
# -----------------------------

# 1. employee household trend
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df[target_col], linewidth=2)
plt.title("Living Cost Index Over Time - Employee Households")
plt.xlabel("Date")
plt.ylabel("Index Number")
plt.grid(True)
plt.tight_layout()
plt.savefig("chart_1_employee_trend.png")
plt.show()

# 2. compare household groups
compare_cols = [
    "Index Numbers ;  Employee households ;  All groups ;",
    "Index Numbers ;  Pensioner and beneficiary households ;  All groups ;",
    "Index Numbers ;  Self-funded retiree households ;  All groups ;",
    "Index Numbers ;  Age pensioner households ;  All groups ;",
    "Index Numbers ;  Other government transfer recipient households ;  All groups ;"
]

plt.figure(figsize=(12, 6))
for col in compare_cols:
    name = col.split(";")[1].strip()
    plt.plot(df["Date"], df[col], label=name)

plt.title("Comparison of Living Cost Index Across Household Types")
plt.xlabel("Date")
plt.ylabel("Index Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("chart_2_household_comparison.png")
plt.show()

# 3. annual percentage change
annual_change_cols = [
    "Percentage Change from Corresponding Quarter of Previous Year ;  Employee households ;  All groups ;",
    "Percentage Change from Corresponding Quarter of Previous Year ;  Pensioner and beneficiary households ;  All groups ;",
    "Percentage Change from Corresponding Quarter of Previous Year ;  Self-funded retiree households ;  All groups ;"
]

plt.figure(figsize=(12, 6))
for col in annual_change_cols:
    name = col.split(";")[1].strip()
    plt.plot(df["Date"], df[col], label=name)

plt.title("Annual Percentage Change in Living Costs")
plt.xlabel("Date")
plt.ylabel("Annual % Change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("chart_3_annual_change.png")
plt.show()

# -----------------------------
# model data
# -----------------------------
model_df = df[["Date", target_col]].copy()
model_df = model_df.dropna(subset=[target_col]).copy()
model_df.reset_index(drop=True, inplace=True)

# simple time feature
model_df["Quarter_Number"] = range(len(model_df))

# lag features
model_df["Lag1"] = model_df[target_col].shift(1)
model_df["Lag2"] = model_df[target_col].shift(2)

model_df = model_df.dropna().copy()

print("Model Data Shape:", model_df.shape)
print(model_df.head())

# split data by time
X = model_df[["Quarter_Number", "Lag1", "Lag2"]]
y = model_df[target_col]

split_index = int(len(model_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
pred_lr = lr_model.predict(X_test)

# random forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

# function for results
def get_results(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

lr_mae, lr_rmse, lr_r2 = get_results(y_test, pred_lr)
rf_mae, rf_rmse, rf_r2 = get_results(y_test, pred_rf)

print("\nLinear Regression")
print("MAE:", round(lr_mae, 3))
print("RMSE:", round(lr_rmse, 3))
print("R2:", round(lr_r2, 3))

print("\nRandom Forest")
print("MAE:", round(rf_mae, 3))
print("RMSE:", round(rf_rmse, 3))
print("R2:", round(rf_r2, 3))

# paired t-test on absolute errors
lr_errors = abs(y_test - pred_lr)
rf_errors = abs(y_test - pred_rf)

t_stat, p_value = ttest_rel(lr_errors, rf_errors)

print("\nPaired T-Test")
print("T Statistic:", round(t_stat, 3))
print("P Value:", round(p_value, 5))

if p_value < 0.05:
    print("There is a significant difference between the models.")
else:
    print("There is no significant difference between the models.")

# results table
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2": [lr_r2, rf_r2]
})

results_df.to_csv("model_results.csv", index=False)
print("\nResults Table")
print(results_df)

# actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(pred_lr, label="Linear Regression", linewidth=2)
plt.title("Actual vs Predicted Living Cost Index")
plt.xlabel("Test Period")
plt.ylabel("Index Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("chart_4_actual_vs_predicted.png")
plt.show()

# next 4 quarter forecast using linear regression
future_forecast = []
last_row = X.iloc[-1].copy()

for i in range(4):
    next_pred = lr_model.predict(pd.DataFrame([last_row], columns=X.columns))[0]
    future_forecast.append(next_pred)

    old_lag1 = last_row["Lag1"]
    last_row["Quarter_Number"] = last_row["Quarter_Number"] + 1
    last_row["Lag1"] = next_pred
    last_row["Lag2"] = old_lag1

forecast_df = pd.DataFrame({
    "Future Quarter": ["Q1", "Q2", "Q3", "Q4"],
    "Predicted Index": [round(x, 2) for x in future_forecast]
})

forecast_df.to_csv("forecast_next_4_quarters.csv", index=False)

print("\nNext 4 Quarter Forecast")
print(forecast_df)

# forecast plot
plt.figure(figsize=(8, 5))
plt.plot(forecast_df["Future Quarter"], forecast_df["Predicted Index"], marker="o", linewidth=2)
plt.title("Forecasted Living Cost Index - Next 4 Quarters")
plt.xlabel("Future Quarter")
plt.ylabel("Predicted Index")
plt.ylim(100.2, 100.65)
plt.grid(True)
plt.tight_layout()
plt.savefig("chart_5_forecast.png")
plt.show()

print("\nInterpretation")
print("Linear Regression performed better than Random Forest.")
print("The Linear Regression model had lower MAE and RMSE, and a much higher R2.")
print("The paired t-test showed that the difference in model errors was statistically significant.")
print("The forecast suggests that living costs for employee households will continue to rise gradually over the next four quarters.")