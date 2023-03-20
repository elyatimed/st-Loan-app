import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
with open('classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# title
st.title('Online P2P Lending Decision')


# Define the input widgets
CreditGrade = st.selectbox('Credit Grade', ('C', 'D', 'B', 'AA', 'HR', 'A', 'E', 'NC'))
EmploymentStatus = st.selectbox('EmploymentStatus', ('Self-employed', 'Not available', 'Full-time', 'Employed', 'Other', 'Not employed', 'Part-time', 'Retired'))
StatedMonthlyIncome =st.number_input("StatedMonthlyIncome", min_value=0, max_value=100000)
MonthlyLoanPayment =st.number_input("MonthlyLoanPayment", min_value=0, max_value=100000)
LoanOriginalAmount =st.number_input("LoanOriginalAmount", min_value=0, max_value=1000000)
BorrowerRate = st.slider('Borrower Rate', 0.0, 1.0, 0.5)
BorrowerAPR = st.slider('BorrowerAPR', 0.0, 1.0, 0.5)
ProsperScore = st.slider('ProsperScore', 1.0, 10.0, 5.0)
ListingCategory = st.slider('ListingCategory (numeric)', 0.0, 20.0, 10.0)
TotalTrades = st.slider('TotalTrades', 0.00,200.00,50.00)
Investors = st.slider('Investors', 1.00,2000.00,1.00)
DebtToIncomeRatio = st.slider('DebtToIncomeRatio',0.00,1000.00,500.00)
Term = st.slider('Term',0.00,100.00,50.00)
BankcardUtilization = st.slider('BankcardUtilization',0.00,100.00,50.00)
data = {'CreditGrade': CreditGrade,
        'Term':Term,
        'BorrowerAPR': BorrowerAPR,
        'BorrowerRate': BorrowerRate ,
        'ProsperScore': ProsperScore ,
        'ListingCategory (numeric)': ListingCategory ,
        'EmploymentStatus': EmploymentStatus,
        'BankcardUtilization': BankcardUtilization ,
        'TotalTrades': TotalTrades,
        'DebtToIncomeRatio': DebtToIncomeRatio,
        'StatedMonthlyIncome':StatedMonthlyIncome,
        'LoanOriginalAmount' : LoanOriginalAmount ,
         'MonthlyLoanPayment': MonthlyLoanPayment,
         'Investors': Investors,       
                }

input_df = pd.DataFrame([data])

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
Loans = pd.read_csv('Clean_data.csv')
df = pd.concat([input_df,Loans],axis=0)
# Encode the categorical columns using OneHotEncoder
encode = ['CreditGrade','EmploymentStatus']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
# Make the prediction using the trained model
df = df.loc[:, ~df.columns.duplicated()]

prediction = model.predict(df)

# Display the prediction
# Add a button to trigger the prediction
if st.button('Predict'):
    # Display the prediction as an expression in a colored box
    if prediction == 1:
         st.markdown(f'<span style="color: green;">{"Good credit standing"}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color: red;"> {"Default risk"} </div>', unsafe_allow_html=True)

