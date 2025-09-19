# TrackMyStack – AI Spending Analyzer & Dashboard

**TrackMyStack** is a personal finance application that helps users understand and optimize their spending. Using Open Banking APIs, AI-driven categorization, and an interactive JavaFX dashboard, the app turns raw transaction data into actionable insights and predictive spending forecasts.

---

## 🌟 Features

- **Bank Transaction Integration**  
  Connects to TrueLayer sandbox to pull account and transaction data.

- **AI/NLP Spending Categorization**  
  Automatically classifies transactions into categories like groceries, subscriptions, income, and entertainment.

- **Interactive Dashboard**  
  JavaFX GUI with charts, monthly trends, and spending summaries.

- **Predictive Insights**  
  Estimates next month’s spending per category based on historical data.

- **Alerts & Notifications**  
  Highlights overspending and unusual transactions.

---

## 🛠️ Tech Stack

- **Java (OOP)** – Core application logic  
- **JavaFX** – Dashboard and data visualization  
- **Python (optional)** – AI/NLP for categorization & predictive analysis  
- **TrueLayer Sandbox API** – Access bank transactions  
- **SQLite / CSV** – Local transaction storage  

---

## 🔒 Security

- API credentials (`CLIENT_ID`, `CLIENT_SECRET`, `REFRESH_TOKEN`) are **never hardcoded**.  
- Use `.env` file for credentials (see `.env.example`).  

---

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/TrackMyStack.git
cd TrackMyStack
