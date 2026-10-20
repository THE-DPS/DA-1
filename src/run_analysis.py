import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "listings.csv.gz"
out_data = ROOT / "data"
figures = ROOT / "figures"
out_data.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
df = pd.read_csv(source, compression="gzip", low_memory=False)
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].replace(r"^\\s*$", np.nan, regex=True)
for c in ["last_scraped","host_since","calendar_last_scraped","first_review","last_review",
          "price_quote_checkin_date","price_quote_checkout_date"]:
    if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")
for c in ["host_is_superhost","host_has_profile_pic","host_identity_verified","has_availability","instant_bookable"]:
    if c in df:
        df[c] = df[c].astype("string").str.lower().map({"t":True,"true":True,"yes":True,"f":False,"false":False,"no":False})
df["price_per_night_usd"] = pd.to_numeric(df.get("price_quote_price_per_night"), errors="coerce")
if "price" in df:
    parsed = pd.to_numeric(df["price"].astype(str).str.replace(r"[$,]","",regex=True), errors="coerce")
    df["price_per_night_usd"] = df["price_per_night_usd"].fillna(parsed)
df = df.drop_duplicates()
if "id" in df: df = df.drop_duplicates(subset=["id"], keep="first")
for c in ["latitude","longitude","accommodates","bathrooms","bedrooms","beds","minimum_nights","maximum_nights",
          "availability_30","availability_60","availability_90","availability_365","number_of_reviews",
          "number_of_reviews_ltm","review_scores_rating","reviews_per_month","host_listings_count",
          "calculated_host_listings_count","estimated_occupancy_l365d","estimated_revenue_l365d","price_per_night_usd"]:
    if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
if "latitude" in df: df.loc[~df.latitude.between(-90,90), "latitude"] = np.nan
if "longitude" in df: df.loc[~df.longitude.between(-180,180), "longitude"] = np.nan
for c in ["accommodates","bedrooms","beds","bathrooms","minimum_nights","maximum_nights","availability_30",
          "availability_60","availability_90","availability_365","number_of_reviews","number_of_reviews_ltm",
          "host_listings_count","calculated_host_listings_count"]:
    if c in df: df.loc[df[c] < 0, c] = np.nan
df.loc[df.price_per_night_usd <= 0, "price_per_night_usd"] = np.nan
q1,q3 = df.price_per_night_usd.quantile([.25,.75]); iqr=q3-q1
df["price_outlier_iqr"] = (df.price_per_night_usd < max(0,q1-1.5*iqr)) | (df.price_per_night_usd > q3+1.5*iqr)
df["host_tenure_years_derived"] = ((df["last_scraped"]-df["host_since"]).dt.days/365.25).clip(lower=0) if "last_scraped" in df and "host_since" in df else np.nan
df.to_csv(out_data/"listings_cleaned.csv", index=False)
df.isna().mean().sort_values(ascending=False).to_csv(ROOT/"missingness_summary.csv")
target="price_per_night_usd"; neighbourhood="neighbourhood_cleansed" if "neighbourhood_cleansed" in df else "neighbourhood"
features=[c for c in ["room_type","property_type",neighbourhood,"accommodates","bathrooms","bedrooms","beds","minimum_nights",
 "maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews",
 "number_of_reviews_ltm","review_scores_rating","host_is_superhost","host_identity_verified","instant_bookable",
 "host_listings_count","host_tenure_years_derived"] if c in df]
data=df[features+[target]].dropna(subset=[target]); X=data[features]; y=data[target]
cats=X.select_dtypes(include=["object","string","bool","category"]).columns.tolist()
nums=[c for c in X.columns if c not in cats]
prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={"median_baseline":Pipeline([("prep",prep),("model",DummyRegressor(strategy="median"))]),
 "random_forest":Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1))])}
results={}
for name,model in models.items():
 model.fit(Xtr,ytr); pred=model.predict(Xte)
 results[name]={"MAE":mean_absolute_error(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"R2":r2_score(yte,pred)}
print("Rows:",len(df),"Columns:",len(df.columns))
print("Model records:",len(data),"Train:",len(Xtr),"Test:",len(Xte))
print(results)
