import pandas as pd
df = pd.DataFrame({
    "Date": pd.date_range("2022-01-01", periods=30),
    "Sales": [100 + i * 5 for i in range(30)],
    "Region": ["East"] * 15 + ["West"] * 15
})
df.to_csv("sample_sales.csv", index=False)
