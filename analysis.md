# Analysis of DeFi Wallet Credit Scores

## **Score Distribution**

The following histogram shows the **distribution of wallet credit scores** across defined ranges (0-100, 100-200, ..., 900-1000):

<img width="1000" height="600" alt="score_distribution" src="https://github.com/user-attachments/assets/d4970320-e980-4099-956b-96988b15c2ec" />

---

## **Behavior Analysis**

### **Wallets in Lower Score Range (0-300)**

- Tend to have:
  - High liquidation rates  
  - Low deposit-to-borrow ratios  
  - Irregular or bot-like transaction patterns  
  - Sparse activity with long gaps between transactions  
  - Low total deposits and repayments compared to borrows

### **Wallets in Higher Score Range (700-1000)**

- Tend to have:
  - Consistent deposits and repayments  
  - High repayment ratios (close to or above 1)  
  - Lower liquidation rates  
  - Diverse asset usage and pool interactions  
  - Active and stable transaction histories with frequent usage

---

**Conclusion:**  
The model effectively differentiates risky wallets (low scores) from reliable, responsible wallets (high scores), aligning with expected DeFi user behavior patterns.

---

