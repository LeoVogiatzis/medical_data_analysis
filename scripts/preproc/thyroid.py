import pandas as pd
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)

df = pd.read_csv("../../medical_data/Thyroid_Sick.csv")

df.dropna(subset=['sex'], inplace=True)

df = df.replace(["f", "t"], [0, 1])
df = df.replace(["F", "M"], [0, 1])

target = pd.DataFrame(df['Class'])
target.replace(["negative", "sick"], [0, 1], inplace=True)

df.drop(["TBG", "Class", "referral source"], axis=1, inplace=True)
df = df.loc[:, ~df.columns.str.endswith("measured")]

imputer = KNNImputer(n_neighbors=5, weights="uniform")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# df.reset_index(drop=True, inplace=True)
# target.reset_index(drop=True, inplace=True)
df = pd.concat([df, target], axis=1)
print(df.head(10))

df.to_csv("../../medical_data/Thyroid_Sick-clean.csv", index=False)
