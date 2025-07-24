import pandas as pd
from decimal import Decimal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
transactions = pd.read_json("user-wallet-transactions.json")

# --- Data Cleaning and Conversion ---

actionData_df=pd.json_normalize(transactions.actionData)
actionData=actionData_df.iloc[:,1:5]

updatedAt_df=pd.json_normalize(transactions.createdAt)
updatedAt_df.rename(columns={"$date":"updated_date"},inplace=True)
updatedAt_df.updated_date=pd.to_datetime(updatedAt_df.updated_date)

createdAt_df=pd.json_normalize(transactions.createdAt)
createdAt_df.rename(columns={"$date":"created_date"},inplace=True)
createdAt_df.created_date=pd.to_datetime(createdAt_df.created_date)

id=pd.json_normalize(transactions._id)
id.rename(columns={"$oid": "id"},inplace=True)

df=pd.concat([id,transactions,actionData,createdAt_df,updatedAt_df],axis=1)
df.rename(columns={"created_date":"createdandupdate_date"},inplace=True)
df=df.drop(["_id","network","protocol","logId","__v","actionData","createdAt","updatedAt","updated_date"], axis=1)

df['amount'] = df['amount'].apply(lambda x: float(Decimal(x)))
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- txHash count per user ---
txHash_count = df.groupby("userWallet")["txHash"].count().reset_index()
txHash_count.rename(columns={"txHash": "txHash_count"}, inplace=True)

# --- unique asset count ---
unique_asset_count = df.groupby("userWallet")["assetSymbol"].nunique().reset_index()
unique_asset_count.rename(columns={"assetSymbol": "unique_asset_count"}, inplace=True)

# --- unique pool count ---
unique_pool_count = df.groupby("userWallet")["poolId"].nunique().reset_index()
unique_pool_count.rename(columns={"poolId": "unique_pool_count"}, inplace=True)

# --- total deposit ---
total_deposit = df[df['action'] == 'deposit'].groupby('userWallet')['amount'].sum().reset_index()
total_deposit.columns = ['userWallet', 'total_deposit']

# --- total borrow ---
total_borrow = df[df['action'] == 'borrow'].groupby('userWallet')['amount'].sum().reset_index()
total_borrow.columns = ['userWallet', 'total_borrow']

# --- deposit to borrow ratio ---
deposit_to_borrow = pd.merge(total_deposit, total_borrow, on='userWallet', how='outer').fillna(0)
deposit_to_borrow['deposit_to_borrow_ratio'] = deposit_to_borrow['total_deposit'] / deposit_to_borrow['total_borrow']
deposit_to_borrow['deposit_to_borrow_ratio'] = deposit_to_borrow['deposit_to_borrow_ratio'].replace([float('inf'), -float('inf')], 0).fillna(0)
deposit_to_borrow_ratio = deposit_to_borrow[['userWallet', 'deposit_to_borrow_ratio']]

# --- total repay ---
total_repay = df[df['action'] == 'repay'].groupby('userWallet')['amount'].sum().reset_index()
total_repay.columns = ['userWallet', 'total_repay']

# --- net borrow ---
net_borrow = pd.merge(total_borrow, total_repay, on='userWallet', how='outer').fillna(0)
net_borrow['net_borrow'] = net_borrow['total_borrow'] - net_borrow['total_repay']
net_borrow = net_borrow[['userWallet', 'net_borrow']]

# --- avg tx amount ---
avg_tx_amount = df.groupby('userWallet')['amount'].mean().reset_index()
avg_tx_amount.columns = ['userWallet', 'avg_tx_amount']

# --- days since last tx ---
latest_tx = df.groupby('userWallet')['timestamp'].max().reset_index()
latest_tx['days_since_last_tx'] = (pd.Timestamp.now() - latest_tx['timestamp']).dt.days
days_since_last_tx = latest_tx[['userWallet', 'days_since_last_tx']]

# --- tx amount std, max, min ---
tx_amount_std = df.groupby('userWallet')['amount'].std().reset_index().fillna(0)
tx_amount_std.columns = ['userWallet', 'tx_amount_std']

tx_amount_max = df.groupby('userWallet')['amount'].max().reset_index()
tx_amount_max.columns = ['userWallet', 'tx_amount_max']

tx_amount_min = df.groupby('userWallet')['amount'].min().reset_index()
tx_amount_min.columns = ['userWallet', 'tx_amount_min']

# --- repayment ratio ---
repayment = pd.merge(total_repay, total_borrow, on='userWallet', how='outer').fillna(0)
repayment['repayment_ratio'] = repayment['total_repay'] / repayment['total_borrow']
repayment['repayment_ratio'] = repayment['repayment_ratio'].replace([float('inf'), -float('inf')], 0).fillna(0)
repayment_ratio = repayment[['userWallet', 'repayment_ratio']]

# --- liquidation rate ---
total_liquidation = df[df['action'] == 'liquidationcall'].groupby('userWallet')['amount'].count().reset_index()
total_liquidation.columns = ['userWallet', 'total_liquidation']

liquidation = pd.merge(total_liquidation, total_borrow, on='userWallet', how='outer').fillna(0)
liquidation['liquidation_rate'] = liquidation['total_liquidation'] / liquidation['total_borrow']
liquidation['liquidation_rate'] = liquidation['liquidation_rate'].replace([float('inf'), -float('inf')], 0).fillna(0)
liquidation_rate = liquidation[['userWallet', 'liquidation_rate']]

# --- user active days ---
tx_dates = df.groupby("userWallet")["timestamp"].agg(["min", "max"]).reset_index()
tx_dates["user_active_days"] = (tx_dates["max"] - tx_dates["min"]).dt.days
user_active_days = tx_dates[['userWallet', 'user_active_days']]

# --- avg time gap ---
data_sorted = df.sort_values(["userWallet", "timestamp"])
data_sorted["time_diff"] = data_sorted.groupby("userWallet")["timestamp"].diff().dt.total_seconds()
avg_time_gap = data_sorted.groupby("userWallet")["time_diff"].mean().fillna(0).reset_index()
avg_time_gap.columns = ['userWallet', 'avg_time_gap']

# -----------------------------
# Final merged feature set
# -----------------------------
final_df = txHash_count.copy()

# Merge step by step for reliability
for feature_df in [
    unique_asset_count,
    unique_pool_count,
    total_deposit,
    total_borrow,
    deposit_to_borrow_ratio,
    total_repay,
    net_borrow,
    avg_tx_amount,
    days_since_last_tx,
    tx_amount_std,
    tx_amount_max,
    tx_amount_min,
    repayment_ratio,
    liquidation_rate,
    user_active_days,
    avg_time_gap
]:
    final_df = final_df.merge(feature_df, on='userWallet', how='left')

final_df = final_df[[
    'userWallet',
    'txHash_count',
    'unique_asset_count',
    'unique_pool_count',
    'total_deposit',
    'total_borrow',
    'deposit_to_borrow_ratio',
    'total_repay',
    'net_borrow',
    'repayment_ratio',
    'liquidation_rate',
    'avg_tx_amount',
    'tx_amount_std',
    'tx_amount_max',
    'tx_amount_min',
    'user_active_days',
    'avg_time_gap',
    'days_since_last_tx'
]]


# Fill NaNs with 0
final_df = final_df.fillna(0)



# Prepare feature matrix
X = final_df.drop(columns=["userWallet"])

# Train Isolation Forest
iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso.fit(X)

# Get anomaly scores
anomaly_scores = iso.decision_function(X)  # Higher is more normal

# Convert to credit scores between 0 and 1000
scaler = MinMaxScaler(feature_range=(0, 1000))
credit_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()

# Add credit scores to dataframe
final_df["credit_score"] = credit_scores

# --- Save model  ---
joblib.dump(scaler, "credit_score_model.joblib")

# View sample
print(final_df[["userWallet", "credit_score"]].head())

