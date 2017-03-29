import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import product
from sklearn.model_selection import StratifiedKFold


def add_features(df):
    fmt = lambda s: s.replace("\u00a0", "").strip().lower()
    df["photo_count"] = df["photos"].apply(len)
    df["street_address"] = df['street_address'].apply(fmt)
    df["display_address"] = df["display_address"].apply(fmt)
    df["desc_wordcount"] = df["description"].apply(len)
    df["pricePerBed"] = df['price'] / df['bedrooms']
    df["pricePerBath"] = df['price'] / df['bathrooms']
    df["pricePerRoom"] = df['price'] / (df['bedrooms'] + df['bathrooms'])
    df["bedPerBath"] = df['bedrooms'] / df['bathrooms']
    df["bedBathDiff"] = df['bedrooms'] - df['bathrooms']
    df["bedBathSum"] = df["bedrooms"] + df['bathrooms']
    df["bedsPerc"] = df["bedrooms"] / (df['bedrooms'] + df['bathrooms'])

    df = df.fillna(-1).replace(np.inf, -1)
    return df


def factorize(df1, df2, column):
    ps = df1[column].append(df2[column])
    factors = ps.factorize()[0]
    df1[column] = factors[:len(df1)]
    df2[column] = factors[len(df1):]
    return df1, df2


def designate_single_observations(df1, df2, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    return df1, df2


def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    if update_df is None: update_df = test_df
    if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    update_df.update(df)
    return

# Load data
X_train = pd.read_json("../input/train.json").sort_values(by="listing_id")
X_test = pd.read_json("../input/test.json").sort_values(by="listing_id")

# Make target integer, one hot encoded, calculate target priors
X_train = X_train.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})
X_train = X_train.join(pd.get_dummies(X_train["interest_level"], prefix="pred").astype(int))
prior_0, prior_1, prior_2 = X_train[["pred_0", "pred_1", "pred_2"]].mean()

# Add common features
X_train = add_features(X_train)
X_test = add_features(X_test)

# Special designation for building_ids, manager_ids, display_address with only 1 observation
for col in ('building_id', 'manager_id', 'display_address'):
    X_train, X_test = designate_single_observations(X_train, X_test, col)

# High-Cardinality Categorical encoding
skf = StratifiedKFold(5)
attributes = product(("building_id", "manager_id"), zip(("pred_1", "pred_2"), (prior_1, prior_2)))
for variable, (target, prior) in attributes:
    hcc_encode(X_train, X_test, variable, target, prior, k=5, r_k=None)
    for train, test in skf.split(np.zeros(len(X_train)), X_train['interest_level']):
        hcc_encode(X_train.iloc[train], X_train.iloc[test], variable, target, prior, k=5, r_k=0.01, update_df=X_train)

# Factorize building_id, display_address, manager_id, street_address
for col in ('building_id', 'display_address', 'manager_id', 'street_address'):
    X_train, X_test = factorize(X_train, X_test, col)

# Create binarized features
fmt = lambda feat: [s.replace("\u00a0", "").strip().lower().replace(" ", "_") for s in feat]  # format features
X_train["features"] = X_train["features"].apply(fmt)
X_test["features"] = X_test["features"].apply(fmt)
features = [f for f_list in list(X_train["features"]) + list(X_test["features"]) for f in f_list]
ps = pd.Series(features)
grouped = ps.groupby(ps).agg(len)
features = grouped[grouped >= 10].index.sort_values().values    # limit to features with >=10 observations
mlb = MultiLabelBinarizer().fit([features])
columns = ['feature_' + s for s in mlb.classes_]
flt = lambda l: [i for i in l if i in mlb.classes_]     # filter out features not present in MultiLabelBinarizer
X_train = X_train.join(pd.DataFrame(data=mlb.transform(X_train["features"].apply(flt)), columns=columns, index=X_train.index))
X_test = X_test.join(pd.DataFrame(data=mlb.transform(X_test["features"].apply(flt)), columns=columns, index=X_test.index))

# Save
"""
X_train = X_train.sort_index(axis=1).sort_values(by="listing_id")
X_test = X_test.sort_index(axis=1).sort_values(by="listing_id")
columns_to_drop = ["photos", "pred_0","pred_1", "pred_2", "description", "features", "created"]
X_train.drop([c for c in X_train.columns if c in columns_to_drop], axis=1).\
    to_csv("data/train_python.csv", index=False, encoding='utf-8')
X_test.drop([c for c in X_test.columns if c in columns_to_drop], axis=1).\
    to_csv("data/test_python.csv", index=False, encoding='utf-8')
"""    