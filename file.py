import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
import json
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify environment variables
required_env_vars = ['TRUELAYER_CLIENT_ID', 'TRUELAYER_CLIENT_SECRET', 'REDIRECT_URI']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class TrackMyStack:
    def __init__(self):
        """Initialize TrackMyStack with API credentials from environment variables"""
        self.token_url = "https://auth.truelayer-sandbox.com/connect/token"
        self.data_api_base = "https://api.truelayer-sandbox.com/data/v1"
        self.client_id = os.getenv('TRUELAYER_CLIENT_ID')
        self.client_secret = os.getenv('TRUELAYER_CLIENT_SECRET')
        self.redirect_uri = os.getenv('REDIRECT_URI')
        self.db_path = 'trackmystack.db'
        self.initialize_database()

    def initialize_database(self):
        """Set up SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables with proper constraints
        c.executescript('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT,
                amount REAL,
                description TEXT,
                merchant_name TEXT,
                timestamp TEXT,
                category TEXT,
                predicted_category TEXT,
                confidence REAL,
                UNIQUE(account_id, amount, description, merchant_name, timestamp)
            );

            CREATE TABLE IF NOT EXISTS spending_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                predicted_amount REAL,
                confidence_interval REAL,
                prediction_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                message TEXT,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()
        conn.close()

    def get_access_token(self):
        """Securely obtain access token using environment variables"""
        refresh_token = os.getenv('REFRESH_TOKEN')
        if refresh_token:
            data = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": refresh_token
            }
        else:
            auth_code = os.getenv('AUTH_CODE')
            if not auth_code:
                raise ValueError("Neither refresh token nor auth code available")
            data = {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "code": auth_code
            }

        resp = requests.post(self.token_url, data=data)
        if resp.status_code == 200:
            resp_json = resp.json()
            if "refresh_token" in resp_json:
                # Update .env file with new refresh token
                self.update_env_file('REFRESH_TOKEN', resp_json['refresh_token'])
            return resp_json['access_token']
        raise Exception(f"Failed to get access token: {resp.text}")

    def predict_next_month_spending(self):
        """Use ML to predict next month's spending per category"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT category, amount, timestamp 
            FROM transactions 
            WHERE amount < 0
        """, conn)
        conn.close()

        if df.empty:
            return {}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['amount'] = df['amount'].abs()

        predictions = {}
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            if len(cat_data) < 2:
                continue

            X = cat_data[['month']]
            y = cat_data['amount']

            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)

            # Predict next month
            next_month = (datetime.now().month % 12) + 1
            pred_amount = model.predict([[next_month]])[0]
            predictions[category] = {
                'amount': round(pred_amount, 2),
                'confidence': round(model.score(X, y) * 100, 2)
            }

        return predictions

    def generate_alerts(self):
        """Generate spending alerts and unusual transaction notifications"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Check for unusual spending patterns
        c.execute("""
            WITH monthly_avg AS (
                SELECT category, AVG(ABS(amount)) as avg_amount
                FROM transactions
                GROUP BY category
            )
            SELECT t.category, t.amount, t.merchant_name, ma.avg_amount
            FROM transactions t
            JOIN monthly_avg ma ON t.category = ma.category
            WHERE ABS(t.amount) > ma.avg_amount * 2
            AND t.timestamp >= date('now', '-7 days')
        """)
        
        unusual_transactions = c.fetchall()
        
        for tx in unusual_transactions:
            alert_msg = f"Unusual spending: {tx[2]} (£{abs(tx[1]):.2f}) in {tx[0]}"
            c.execute("""
                INSERT INTO alerts (type, message, severity)
                VALUES (?, ?, ?)
            """, ('unusual_transaction', alert_msg, 'high'))

        conn.commit()
        conn.close()

    def update_env_file(self, key, value):
        """Safely update .env file with new values"""
        env_path = '.env'
        if os.path.exists(env_path):
            with open(env_path, 'r') as file:
                lines = file.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"{key}={value}\n")

            with open(env_path, 'w') as file:
                file.writelines(lines)

def main():
    """Main application entry point"""
    try:
        logger.info("Initializing TrackMyStack...")
        app = TrackMyStack()
        
        logger.info("Getting access token...")
        access_token = app.get_access_token()
        logger.info("Access token obtained successfully")
        
        # Fetch and store transactions
        logger.info("Fetching transactions...")
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # First get accounts
        accounts_response = requests.get(
            f"{app.data_api_base}/accounts",
            headers=headers
        )
        accounts = accounts_response.json().get('results', [])
        
        # Then fetch transactions for each account
        for account in accounts:
            account_id = account['account_id']
            transactions_response = requests.get(
                f"{app.data_api_base}/accounts/{account_id}/transactions",
                headers=headers
            )
            transactions = transactions_response.json().get('results', [])
            
            # Store transactions in database
            conn = sqlite3.connect(app.db_path)
            c = conn.cursor()
            for tx in transactions:
                try:
                    c.execute("""
                        INSERT OR IGNORE INTO transactions 
                        (account_id, amount, description, merchant_name, timestamp, category)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        account_id,
                        tx.get('amount', 0),
                        tx.get('description', ''),
                        tx.get('merchant_name', ''),
                        tx.get('timestamp', ''),
                        # Simple category logic - you might want to enhance this
                        'UNCATEGORIZED'
                    ))
                except sqlite3.Error as e:
                    logger.error(f"Error storing transaction: {e}")
            conn.commit()
            conn.close()
            
        logger.info(f"Fetched and stored transactions for {len(accounts)} accounts")
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = app.predict_next_month_spending()
        if predictions:
            print("\n📊 Next Month's Spending Predictions:")
            for category, pred in predictions.items():
                print(f"{category}: £{pred['amount']} (Confidence: {pred['confidence']}%)")
        else:
            logger.warning("No predictions generated - insufficient data")

        # Generate alerts
        logger.info("Generating alerts...")
        app.generate_alerts()
        
        print("\n✅ Analysis complete! Data ready for JavaFX dashboard.")
        
    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        print(f"❌ Configuration error: {str(ve)}")
    except requests.exceptions.RequestException as re:
        logger.error(f"API request failed: {str(re)}")
        print(f"❌ API request failed: {str(re)}")
    except sqlite3.Error as se:
        logger.error(f"Database error: {str(se)}")
        print(f"❌ Database error: {str(se)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()