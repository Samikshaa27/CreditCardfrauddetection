import streamlit as st
import pickle
import numpy as np

# Load models
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

FEATURE_ORDER = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                 'V11','V12','V13','V14','V15','V16','V17','V18','V19',
                 'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

NAME_TO_V = {
    'merchant_category_score': 'V1',
    'transaction_velocity': 'V2',
    'spending_pattern': 'V3',
    'card_usage_frequency': 'V4',
    'geographic_risk_score': 'V5',
    'billing_address_match': 'V6',
    'online_transaction_flag': 'V7',
    'device_trust_score': 'V8',
    'time_since_last_txn': 'V9',
    'account_age_score': 'V10',
    'daily_spending_limit_usage': 'V11',
    'international_txn_flag': 'V12',
    'merchant_reputation': 'V13',
    'card_present_flag': 'V14',
    'decline_rate': 'V15',
    'chargeback_history': 'V16',
    'unusual_hour_flag': 'V17',
    'ip_risk_score': 'V18',
    'multiple_cards_flag': 'V19',
    'transaction_frequency': 'V20',
    'user_behavior_score': 'V21',
    'browser_risk_score': 'V22',
    'email_mismatch_score': 'V23',
    'phone_verification_score': 'V24',
    'address_verification_score': 'V25',
    'velocity_check_score': 'V26',
    'fraud_model_score_a': 'V27',
    'fraud_model_score_b': 'V28',
    'transaction_amount': 'Amount'
}

st.title("🛡️ Credit Card Fraud Detection")

# Inputs
transaction_amount = st.number_input("Transaction Amount", 0.0, 100000.0, 150.0)
merchant_category_score = st.slider("Merchant Category Score", -5.0, 5.0, 0.0)
transaction_velocity = st.slider("Transaction Velocity", -5.0, 5.0, 0.0)
card_usage_frequency = st.slider("Card Usage Frequency", -5.0, 5.0, 0.0)

geographic_risk_score = st.slider("Geographic Risk Score", -5.0, 5.0, 0.0)
international_txn_flag = st.selectbox("International Transaction", [-2, 2])
card_present_flag = st.selectbox("Card Present", [2, -2])
unusual_hour_flag = st.selectbox("Unusual Hour", [-1, 3])

account_age_score = st.slider("Account Age Score", -5.0, 5.0, 0.0)
chargeback_history = st.selectbox("Chargeback History", [-1, 2])
ip_risk_score = st.slider("IP Risk Score", -5.0, 5.0, 0.0)
device_trust_score = st.slider("Device Trust Score", -5.0, 5.0, 0.0)
address_verification_score = st.slider("Address Verification Score", -5.0, 5.0, 0.0)
spending_pattern = st.slider("Spending Pattern", -5.0, 5.0, 0.0)

# Button
if st.button("🔍 Analyze Transaction"):

    input_data = {
        'merchant_category_score': merchant_category_score,
        'transaction_velocity': transaction_velocity,
        'spending_pattern': spending_pattern,
        'card_usage_frequency': card_usage_frequency,
        'geographic_risk_score': geographic_risk_score,
        'billing_address_match': 0,
        'online_transaction_flag': 0,
        'device_trust_score': device_trust_score,
        'time_since_last_txn': 0,
        'account_age_score': account_age_score,
        'daily_spending_limit_usage': 0,
        'international_txn_flag': international_txn_flag,
        'merchant_reputation': 0,
        'card_present_flag': card_present_flag,
        'decline_rate': 0,
        'chargeback_history': chargeback_history,
        'unusual_hour_flag': unusual_hour_flag,
        'ip_risk_score': ip_risk_score,
        'multiple_cards_flag': 0,
        'transaction_frequency': 0,
        'user_behavior_score': 0,
        'browser_risk_score': 0,
        'email_mismatch_score': 0,
        'phone_verification_score': 0,
        'address_verification_score': address_verification_score,
        'velocity_check_score': 0,
        'fraud_model_score_a': 0,
        'fraud_model_score_b': 0,
        'transaction_amount': transaction_amount
    }

    row = {}
    for k, v in NAME_TO_V.items():
        row[v] = float(input_data.get(k, 0))

    row['Amount'] = scaler.transform([[row['Amount']]])[0][0]

    features = np.array([[row[f] for f in FEATURE_ORDER]])

    lr_prob = lr_model.predict_proba(features)[0][1]
    rf_prob = rf_model.predict_proba(features)[0][1]
    avg_prob = (lr_prob + rf_prob) / 2

    st.subheader("📊 Result")
    st.metric("Fraud Probability", f"{round(avg_prob*100,1)}%")

    if avg_prob > 0.5:
        st.error("⚠️ FRAUD DETECTED")
    else:
        st.success("✅ LEGITIMATE")
