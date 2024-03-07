import pickle
import streamlit as st
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title('Customer Personality Analysis')

input_values =  []

birth_year =[''] + [i for i in range(1900,2025)]
select_birth_year = st.selectbox('Year of Birth', birth_year)
income = st.number_input('Income',step=1,format='%d')
Kidhome = st.number_input('No of Kids in Home',step=1,format='%d')
Teenhome = st.number_input('No of Teens in Home',step=1,format='%d')
Dt_Customer = st.date_input('Date of customers enrollment with the company',min_value=datetime(1900, 1, 1)).year
Recency = st.number_input('Recency',step=1,format='%d')
MntWines = st.number_input('Amount spend on Wine',step=1,format='%d')
MntFruits = st.number_input('Amount spend on Fruits',step=1,format='%d')
MntMeatProducts = st.number_input('Amount spend on Meat Products',step=1,format='%d')
MntFishProducts = st.number_input('Amount spend on Fish Products',step=1,format='%d')
MntSweetProducts = st.number_input('Amount spend on Sweet Products',step=1,format='%d')
MntGoldProds = st.number_input('Amount spend on Gold Products',step=1,format='%d')
NumDealsPurchases = st.number_input('No of Deals Purchases',step=1,format='%d')
NumWebPurchases = st.number_input('No of web purchases',step=1,format='%d')
NumCatalogPurchases = st.number_input('No of catalog purchases',step=1,format='%d')
NumStorePurchases = st.number_input('No of stores purchases',step=1,format='%d')
NumWebVisitsMonth = st.number_input('No of web visits per Month',step=1,format='%d')


Cmp_Acc = {'AcceptedCmp3':0,'AcceptedCmp4':0,'AcceptedCmp5':0,'AcceptedCmp1':0,'AcceptedCmp2':0}
Education ={'Basic':0,'Graduation':0,'Master':0,'PhD': 0}
Martial_status = {'Alone':0,'Divorced':0,'Married':0,'Single':0,'Together':0,'Widow':0,'YOLO':0}

Martial_status_list = ['','Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO']
Education_list = ['','Graduation', 'PhD', 'Master', 'Basic', '2n Cycle']
Cmp_Accepted = st.selectbox('Accept Cmp',["None"] + list(Cmp_Acc.keys()))
select_Education = st.selectbox('Education', Education_list)
select_martial_status = st.selectbox('Martial Status', Martial_status_list)


Submit = st.button('Submit')
if Submit:
    if select_birth_year:
        input_values.append(select_birth_year)
    for i in [income,Kidhome, Teenhome,Recency, MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth]:
        input_values.append(i)
    if Cmp_Accepted:
        if Cmp_Accepted != 'None':
          Cmp_Acc[Cmp_Accepted] = 1
          input_values = input_values + list(Cmp_Acc.values())
        else:
            input_values = input_values + list(Cmp_Acc.values())

    input_values.append(Dt_Customer - select_birth_year) # Age
    input_values.append(MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds) # Total Spending
    if select_Education:
        if select_Education != '2n Cycle':
           Education[select_Education] = 1
           input_values = input_values + list(Education.values())
        else:
            input_values = input_values + list(Education.values())
    if select_martial_status:
        if select_martial_status != 'Absurd':
            Martial_status[select_martial_status] = 1
            input_values = input_values + list(Martial_status.values())
        else:
            input_values = input_values + list(Martial_status.values())

    out_put = list(model.predict([input_values]))

    st.markdown(f"**Customer Allocated to Cluster - {out_put[0]}**")

    df = pd.read_csv('Customer_data.csv')
    df_filtered = df[df['Cluster'] == out_put[0]]

    # Columns to plot
    columns_to_plot = df_filtered.drop(['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Response', 'Complain','Cluster'],
                                       axis=1).columns

    # Number of columns per row
    cols_per_row = 2

    # Iterate over columns for plotting
    for i in range(0, len(columns_to_plot), cols_per_row):
        # Create columns for each plot
        cols = st.columns(cols_per_row)
        # Iterate over columns in the current row
        for j, c in enumerate(columns_to_plot[i:i + cols_per_row]):
            # Create a FacetGrid for each column based on 'Cluster'
            grid = sns.FacetGrid(df_filtered, col='Cluster', height=6)
            # Map a histogram for the current column
            grid = grid.map(plt.hist, c)
            # Display the plot in the current column
            with cols[j]:
                st.pyplot(grid) 


# streamlit run Customer_campain_str.py

