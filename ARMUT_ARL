# ASSOCIATION RULE LEARNING
#Bussines Problem
#Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
# It provides easy access to services such as cleaning, modification and transportation with a few touches on your computer or smart phone.
#It is desired to create a product recommendation system with Association Rule Learning by using the data set containing the services and categories that the users have purchased.

#Variables:
# UserId : CustomerID
# ServiceId: They are anonymized services belonging to each category. (Example: Upholstery washing service under cleaning) A ServiceId can be foundunder different categories and refers to different services under different categories.
# CategoryId: They are anonymized categories. (Example: Cleaning, shipping, as follows)
# CreateDate: The date of service which was purchased.

# !pip install mlxtend
import pandas as pd
import datetime
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

##################### Task 1 - Prepare data #####################################
# Step 1: load armut_data.csv

df_ = pd.read_csv("WEEK_5/Ödevler/Armut_ARL/armut_data.csv")
df = df_.copy()
df.head()

def check_df(dataframe, head=5):
    print("############### shape #############")
    print(dataframe.shape)
    print("############### types #############")
    print(dataframe.dtypes)
    print("############### head #############")
    print(dataframe.head())
    print("############### tail #############")
    print(dataframe.tail())
    print("############### NA #############")
    print(dataframe.isnull().sum())
    print("############### Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Step 2: Combine ServiceID and CategoryID
df['Hizmet'] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

df.head()

# Step 3:  The data set consists of the date and time the services are received, there is no basket definition (invoice, etc.).
# In order to apply the Association Rule Learning, the definition of basket (invoice, etc.) must be created.
# Here, the definition of basket is the services that each customer receives on a monthly basis.
# For example; Basket consisting of 4_5, 48_5, 6_7, 47_7 services received by the customer with ID number 25446 in the 8th month of 2017; 17_5, 14_7 Services received in the 9th month of 2017 represent another basket.
# Baskets must be identified by a unique ID. To do this I will first create a new date variable that contains only the year and month.
# Then I will combine UserID and the new date variable with "_" and assign it to a new variable called ID.

df['New_Date'] = pd.to_datetime(df['CreateDate'],format='%Y-%m').dt.to_period('M')

df["SepetID"] = [str(i[0]) + "_" + str(i[5]) for i in df.values]

##################### Task 2 - Create ASSOCIATION RULE LEARNING and give an advice#####################################

# Step 1 : create a dataframe so as to use ARL

df_pivot = df.groupby(["SepetID", "Hizmet"]) \
    ["Hizmet"].count(). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

df_pivot.columns

# Step 2: create ARL

apriori_alg = apriori(df_pivot, min_support=0.01, use_colnames=True,  low_memory=True)
apriori_alg.sort_values("support", ascending=False)

rules = association_rules(apriori_alg,
                          metric="support",
                          min_threshold=0.01)

rules.head()

# Step 3: To suggest a service to a user who has received the 2_0 service in the last 1 month using the arl suggestion Function.
sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, service in enumerate(sorted_rules["antecedents"]):
    for j in list(service): # product'ı listeye çeviriyorum, içerisinde gezebilmek için, çünkü 1 tanesini seçeceğiz. Ama normalde 2 tane var ise tek bir ürün olarak görmeliyiz.
        if j == "2_0":
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

def arl_recommender(rules_df, service_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == service_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0", 4)
