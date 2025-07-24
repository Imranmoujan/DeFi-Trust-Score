# DeFi-Trust-Score

## Problem Statement

Develop a robust machine learning model to assign a **credit score between 0 and 1000** to each wallet based solely on historical transaction behavior from the Aave V2 protocol.

**Higher scores → Reliable users**  
**Lower scores → Risky, bot-like, or exploitative behavior**

---

## **Dataset**

The sample dataset used for training and analysis:

[Download Dataset (JSON, Google Drive)](https://drive.google.com/file/d/1VFHrvj0mkaUg_hdBD5PXiT18aoOBkudy/view?usp=sharing)

---

## **Method Chosen**

We used **Isolation Forest**, an unsupervised anomaly detection algorithm, to analyze wallet behaviors without requiring labeled data.

### **Why Isolation Forest?**

- No ground truth credit score labels available  
- Detects anomalous (risky) vs. normal (reliable) usage patterns efficiently  
- Scales well to 100K+ records

---

## **Architecture Overview**



  <div align="center">

Raw JSON Data  
↓  
Data Cleaning & Normalization  
↓  
Feature Engineering  
↓  
Isolation Forest Model Training  
↓  
Anomaly Scores (Higher = Normal)  
↓  
MinMax Scaling (0–1000)  
↓  
Final Credit Scores per Wallet

</div>

---

##   **Feature Engineering Pipeline**

Extracted features include:

- **Transaction Behavior**
  - Total transactions (`txHash_count`)
  - Unique assets & pools interacted with
  - Total deposit, borrow, and repay amounts
  - Deposit-to-borrow ratio
  - Net borrow amount

- **Transaction Amount Statistics**
  - Average, min, max, std deviation

- **Repayment & Liquidation Behavior**
  - Repayment ratio (repaid / borrowed)
  - Liquidation rate (liquidation calls / borrowed)

- **Activity Metrics**
  - Active days (first to last transaction)
  - Average time gap between transactions
  - Days since last transaction

These features comprehensively describe wallet reliability, diversity, and risk behavior.

---

##  **Modeling Approach**

| Step | Details |
|---|---|
| **Model** | Isolation Forest (`sklearn`) |
| **Features** | All engineered wallet features |
| **Output** | Anomaly scores (higher = normal) |
| **Scaling** | MinMaxScaler mapped anomaly scores to **credit scores between 0 and 1000** |

---
