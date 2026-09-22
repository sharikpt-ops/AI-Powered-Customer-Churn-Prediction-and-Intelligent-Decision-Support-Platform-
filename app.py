from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import sqlite3
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Flash messages ke liye required hai

DB_PATH = os.path.join('database', 'churn_platform.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Database Table Auto-Creation if not exists
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS churn_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenure INTEGER,
            monthly_charges REAL,
            tickets INTEGER,
            churn_probability REAL,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Top Navbar Redirect
@app.route('/')
def index():
    return redirect(url_for('data_quality'))

@app.route('/rule-manager')
def rule_manager():
    return render_template('dashboard.html', module='Rule Manager')

# 1. Data Quality Check & Full Customer Dataset View
@app.route('/data-quality')
def data_quality():
    try:
        conn = get_db_connection()
        total_customers = conn.execute('SELECT COUNT(*) FROM customer_churn').fetchone()[0]
        high_risk_count = conn.execute("SELECT COUNT(*) FROM customer_churn WHERE Churned='Yes' OR Previous_Churn_Risk_Score > 0.5").fetchone()[0]
        avg_spend = conn.execute('SELECT AVG(Monthly_Spend_INR) FROM customer_churn').fetchone()[0] or 0
        
        # All 18 columns fetch karne ke liye
        all_customers = conn.execute('SELECT * FROM customer_churn LIMIT 100').fetchall()
        column_names = all_customers[0].keys() if all_customers else []
        conn.close()
    except Exception:
        total_customers, high_risk_count, avg_spend = 0, 0, 0
        all_customers, column_names = [], []
    
    stats = {
        'total_customers': total_customers,
        'high_risk': high_risk_count,
        'avg_spend': round(avg_spend, 2)
    }
    return render_template('dashboard.html', 
                           module='Data Quality Overview', 
                           stats=stats, 
                           all_customers=all_customers,
                           column_names=column_names)

# 2. Single Batch File Upload (Excel / CSV) - Duplicate removed
@app.route('/batch-upload', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Kripya ek valid Excel ya CSV file select karein!', 'danger')
            return redirect(url_for('batch_upload'))
        
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                flash('Sirf .csv, .xls, ya .xlsx files hi allowed hain!', 'danger')
                return redirect(url_for('batch_upload'))
            
            os.makedirs('database', exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            df.to_sql('customer_churn', conn, if_exists='replace', index=False)
            conn.close()

            flash(f'Successfully uploaded! {len(df)} records aur {len(df.columns)} columns database me save ho gaye.', 'success')
            return redirect(url_for('data_quality'))
            
        except Exception as e:
            flash(f'Upload me error aaya: {str(e)}', 'danger')
            return redirect(url_for('batch_upload'))

    return render_template('dashboard.html', module='Batch CSV Upload')

# 3. EDA & Patterns
@app.route('/eda-patterns')
def eda_patterns():
    return render_template('dashboard.html', module='EDA & Churn Patterns Analysis')

# 4. Feature Engineering
@app.route('/feature-engineering')
def feature_engineering():
    return render_template('dashboard.html', module='Feature Engineering & Leakage Audit')

# 5. Model Leaderboard / Analytics
@app.route('/analytics')
@app.route('/model-comparison')
def model_comparison():
    models_metrics = [
        {'model': 'XGBoost Classifier', 'val_auc': 0.89, 'test_auc': 0.87, 'f1_score': 0.82},
        {'model': 'Random Forest', 'val_auc': 0.86, 'test_auc': 0.84, 'f1_score': 0.79},
        {'model': 'Logistic Regression', 'val_auc': 0.75, 'test_auc': 0.74, 'f1_score': 0.68}
    ]
    return render_template('dashboard.html', module='Model Performance & Leaderboard', metrics=models_metrics)

# 6. Explainability & Fairness
@app.route('/explainability')
def explainability():
    return render_template('dashboard.html', module='Explainable AI (SHAP) & Fairness Audits')

# 7. Predict & Drift Engine
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    conn = get_db_connection()
    prediction_result = None

    if request.method == 'POST':
        data = request.form
        tenure = int(data.get('tenure', 0))
        monthly_charges = float(data.get('monthly_charges', 0))
        tickets = int(data.get('tickets', 0))
        
        prob = min(0.1 + (tickets * 0.15) + (1.0 / (tenure + 1)), 0.99)
        risk_level = "High Churn Risk" if prob > 0.6 else ("Medium Churn Risk" if prob > 0.3 else "Low Churn Risk")
        prob_percentage = round(prob * 100, 2)

        conn.execute('''
            INSERT INTO churn_predictions (tenure, monthly_charges, tickets, churn_probability, risk_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (tenure, monthly_charges, tickets, prob_percentage, risk_level))
        conn.commit()

        prediction_result = {
            'prob': prob_percentage,
            'risk': risk_level
        }

    recent_predictions = conn.execute('SELECT * FROM churn_predictions ORDER BY id DESC LIMIT 10').fetchall()
    conn.close()

    return render_template(
        'dashboard.html', 
        module='Live Churn Prediction', 
        prediction=prediction_result,
        recent_predictions=recent_predictions
    )

# 8. Recommendations & DSS
@app.route('/recommendations')
def recommendations():
    high_risk_customers = []
    try:
        conn = get_db_connection()
        # High Risk Customers fetch karein (Churned = 'Yes' OR Previous_Churn_Risk_Score > 0.4)
        raw_rows = conn.execute("""
            SELECT * FROM customer_churn 
            WHERE Churned='Yes' OR Previous_Churn_Risk_Score > 0.4 
            LIMIT 50
        """).fetchall()
        conn.close()

        for row in raw_rows:
            # Row ko dictionary me convert karein
            c = dict(row)
            
            # 1. Dynamic Risk Category Logic
            score = c.get('Previous_Churn_Risk_Score', 0) or 0
            if c.get('Churned') == 'Yes' or score >= 0.7:
                risk_cat = "High Risk"
            elif score >= 0.4:
                risk_cat = "Medium Risk"
            else:
                risk_cat = "Low Risk"

            # 2. Dynamic Top Risk Factor Logic
            tickets = c.get('Support_Tickets_90D', 0) or 0
            failures = c.get('Payment_Failures_90D', 0) or 0
            contract = c.get('Contract_Type', '') or ''
            
            if tickets >= 3:
                risk_factor = f"High Support Tickets ({tickets} Tickets in 90D)"
            elif failures >= 2:
                risk_factor = f"Payment Failures ({failures} Failures)"
            elif contract == 'Monthly':
                risk_factor = "No Long-term Contract (Monthly Plan)"
            else:
                risk_factor = "Low Session Activity / Engagement"

            # 3. Dynamic Action Recommendation Logic
            if tickets >= 3:
                action = "Assign Dedicated Account Manager & Priority Support"
            elif failures >= 2:
                action = "Offer Auto-Pay Discount & Payment Method Assistance"
            elif contract == 'Monthly':
                action = "Propose 1-Year Annual Subscription with 15% Discount"
            else:
                action = "Send Re-engagement Promo & App Feature Onboarding"

            # Derived fields append karein
            c['Risk_Category'] = risk_cat
            c['Top_Risk_Factor'] = risk_factor
            c['Recommended_Action'] = action
            
            high_risk_customers.append(c)

    except Exception as e:
        print("Recommendations Error:", e)
        high_risk_customers = []

    return render_template('dashboard.html', 
                           module='Decision Support & Intervention Plan', 
                           high_risk_customers=high_risk_customers)

if __name__ == '__main__':
    app.run(debug=True, port=5000)