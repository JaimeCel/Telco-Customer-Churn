import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Churn Predictor', layout='centered')
@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

model_data = load_model()
pipe = model_data['pipeline']
threshold = model_data['threshold']

st.title('Telco Churn Predictor')
st.caption('Predicts whether a customer will churn based on their profile.')

with st.form('prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Account')
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        monthly = st.number_input('Monthly Charges ($)', 18.0, 120.0, 65.0, step=0.5)
        payment = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        paperless = st.selectbox('Paperless Billing', ['Yes', 'No'])

    with col2:
        st.subheader('Profile')
        senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['Yes', 'No'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

        st.subheader('Services')
        internet = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
        security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

    submitted = st.form_submit_button('Predict churn risk', use_container_width=True)

if submitted:
    if internet == 'No':
        security = backup = device = support = tv = movies = 'No internet service'

    input_df = pd.DataFrame([{
        'SeniorCitizen':   1 if senior == 'Yes' else 0,
        'Partner':         partner,
        'Dependents':      dependents,
        'tenure':          tenure,
        'InternetService': internet,
        'OnlineSecurity':  security,
        'OnlineBackup':    backup,
        'DeviceProtection': device,
        'TechSupport':     support,
        'StreamingTV':     tv,
        'StreamingMovies': movies,
        'Contract':        contract,
        'PaperlessBilling': paperless,
        'PaymentMethod':   payment,
        'MonthlyCharges':  monthly,
    }])

    prob = pipe.predict_proba(input_df)[0][1]
    churns = prob >= threshold

    st.divider()
    if churns:
        st.error('Churn: Yes')
    else:
        st.success('Churn: No')
    st.caption(f'Churn probability: {prob:.1%}  |  Decision threshold: {threshold:.1%}')

