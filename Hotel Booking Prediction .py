#!/usr/bin/env python
# coding: utf-8

# # Problem Statement 
#    :- will the guest going to cancel hotel reservation?
#    
#  Objective :- Predict the probabilty of booking cancellation 
#  

# In[5]:


# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib.inline', '')


# In[6]:


# Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[7]:


# Import Datasets
df = pd.read_csv("hotel_booking.csv")


# In[8]:


df.shape


# In[9]:


df.head()


# # Data Cleaning

# In[10]:


# Handling Missing Values

df.isnull().sum()


# df.isnull().mean()

# In[11]:


df.isnull().mean()


# In[12]:


df.isnull().mean().sort_values(ascending = False)


# In[14]:


df.dtypes


# In[15]:


# df.isnull() checks which values in a DataFrame
# .mean() calculates the mean of each column in the resulting DataFrame
# .sort_values(ascending=False) sorts the resulting Series in descending order.
#[0:6] returns the first 6 elements of the sorted Series.
# *100 multiplies each element in the resulting Series by 100, to express the percentage of null values in each column.


# In[16]:


(df.isnull().mean().sort_values(ascending = False)[0:6])*100

* cheking unique value of the columns
# In[17]:


df.children.value_counts()


# In[18]:


df.country.value_counts()[:10]


# In[19]:


df.agent.value_counts()


# In[20]:


df.head(10)


# In[21]:


(df.isnull().mean().sort_values(ascending=False)[:6])*100

* As we have almost 94% null values in "company" column We can remove the column.
* Agent column with 13.6% of null values as this feature is travel agency Id and these values are unique and we cannot impute Id by mean, median or mode. Therefore, missing data for "Agent" can be filled by 0.
* Children column had very low null values can be filled with mode (most repeated count). 
* Country column had very low null values can be filled with mode (most repeated country).
# In[22]:


# Handling null values

df["children"].fillna(df.children.mode()[0], inplace= True) # The [0] index selects the first mode value from the resulting Series.
df["country"].fillna(df.country.mode()[0], inplace = True)   # inplace=True modifies the "children" column of the DataFrame df in place, without creating a new DataFrame.
df["agent"].fillna(0, inplace = True               # As Agent column is IDs had to fill with zero for NaN values.
df["children"] = df["children"].astype("int64")    # convert children column data types as int64

# Here we replaced null of children and country column with the mode values. Mode value is the most repeated value of the column.
# In[23]:


# Dropping company columns which has 94% missing values
df.drop(["company"], axis = 1, inplace = True)


# In[24]:


df.shape


# # Removing redundant features

# In[25]:


# rows which has 0 in all 3 guests types which means 0 guests. # a(room number) = 0, b(room number) = 0 , c(room number) = 0
zeroguest = (df["children"]+df["adults"]+df["babies"]==0) # a+b+c = 0 == 0 = True 1. (1,2,3,4,5), 2. (1,2,3,4,5)
df[zeroguest]


# In[26]:


# we can remove rows which had zero guest in 180 rows
# dropping rows which had 0 guests

df.drop(df[zeroguest].index, inplace = True)


# In[27]:


df.shape


# In[28]:


# checking of duplicates
df.duplicated().sum()


# In[29]:


df.shape


# In[30]:


# Remove Name, email and Phone number which is not required for model -> columns

df.drop(["name","email","phone-number","credit_card"], inplace= True, axis=1)
df


# In[31]:


df.shape


# # DataFrame Exploration 
# 

# In[32]:


df.head()


# In[33]:


df.describe() # Display summary statistics


# In[35]:


df.info()


# # Visualization
 # Guest country analysis
# In[36]:


# create a new dataframe with country and total counts

cust_country_data = df["country"].value_counts().reset_index()
cust_country_data.columns=["country", "Total_guests"]
cust_country_data

* **Extracting Country names from the country column which has code represented in the ISO 3155–3:2013 format**
* **Using Module pycountry with Alpha 3 which is IS0 3155-3:2013 Format**
# In[37]:


get_ipython().system('pip install pycountry')


# In[38]:


# Function to get country name from country code
    
import pycountry
def get_country(n):
    country = pycountry.countries.get(alpha_3 = n)
    if country:
        return country.name
    else:
        return n


# In[39]:


## Creating new dataframe for visualization

# create a list from the customer country data table
lst = cust_country_data.country.to_list()

# create a new columns with new countries

country_name = [get_country(name) for name in lst]
cust_country_data["country_name"] = country_name
cust_country_data.head()


# In[47]:


# Create List of top 12
name = cust_country_data["country_name"].head(12)
total = cust_country_data["Total_guests"].head(12)

## Plot Bar chart of countries
sns.set_style("whitegrid")
plt.subplots(figsize=(13,8))

sns.barplot(x= name[0:10], y = total[0:10])
plt.xlabel("countries", weight = "bold")
plt.ylabel("Total Bookings", weight = "bold")
plt.grid(alpha = 0.5, axis = "y")


# In[40]:


## Plot map chart for all customer countries PLOTLY LIBRAYR
import plotly.express as px

cust_map = px.choropleth(cust_country_data,
                        locations=cust_country_data["country"],
                        color = cust_country_data["Total_guests"],
                        hover_name = cust_country_data["country"],
                        labels = {"Total_guests":"Total_Bookings"},
                        color_continuous_scale = "Turbo")
cust_map.update_layout(margin = {"r":0, "t":0, "l":0, "b":0})
import plotly.graph_objects as go
fig_widget = go.FigureWidget(cust_map)
cust_map


# In[41]:


## Booking count based on Hotel Type ## 
# Count of booking based on type of hotel (Resort, City Hotel)

df.groupby(["is_canceled"])["hotel"].value_counts()  # Zero = NOT CANCELLED AND One = CANCELLED


# In[43]:


# Plot barchart for both hotel type


plt.subplots(figsize=(13,7))
sns.set_style("whitegrid")
sns.countplot(x="hotel", hue="is_canceled", data = df, ec = "black")
plt.title("Booking Cancellations by Hotels types", weight = "bold", fontsize = 15)
plt.ylabel("Booking Count", weight = "bold")
plt.xlabel("Hotel Type", weight = "bold")
plt.grid(alpha = 0.5, axis = "y")
plt.show()

## Legend : Blue = cheked in and Orange = cancelled

The abobe graphs shows city hotels had more booking and cancellation received than Resort Hotel type.
# In[44]:


## Check In Vs Cancellation ##
# percetage of cancelled and not cancelled

percentage = df.is_canceled.value_counts(normalize=True)*100
pielabels = ["Checked-in", "Canceled"]


# In[45]:


# Plot PieChart with Ploty library

f1 = px.pie(values = percentage, names = pielabels, title = "percentage of check-ins Vs cancellation")
f1.update_traces(textposition = "inside", textinfo = "percent+label")
f1.update_layout(margin = {"r":50, "t":50, "l":50, "b":50})
f1.show()

The above pie charts shows checked in = 63% & where as canceled = 37% 
# In[46]:


## Which month is most occupied and least occupied with booking at the hotel?? ##
# Exact Arrival month and know its booking status

# Group by cancellation status and Months

month_data = df.groupby(["is_canceled", "arrival_date_month"])["arrival_date_month"].value_counts()
month_data


# In[48]:


# Group by cancellation status and Months

month_data = df.groupby(["is_canceled", "arrival_date_month"])["arrival_date_month"].value_counts()
month_data


# In[54]:


# Month order to arrange the barchart
months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
df1 = df.copy() # create a copy from main df
# Change values for plotting according to data description
df1["is_canceled"]= np.where(df1["is_canceled"]==0,"Checked-in","Booking Canceled")

# Plot Barchart for month and Bookings
plt.subplots(figsize=(13,6))
sns.set_style("whitegrid")
sns.countplot(x="arrival_date_month",hue="is_canceled",data= df1, order=months_order,ec = "black", )
plt.title("Cancellations and Check-ins Month wise", fontsize=15)
plt.ylabel("Bookings Count", weight = "bold")
plt.xlabel("Months", weight= "bold")
plt.legend(loc="upper right")
plt.grid(alpha = 0.5,axis = "y")
plt.show()

The above Chart shows:-
 -August is the most busiest month in terms of bookings.
 -January is the lowest bookings month.
# In[55]:


## Which year is most occupied and least occupied with booking at the hotel

sns.set_style("whitegrid")
plt.subplots(figsize=(13,6))
sns.countplot(x="arrival_date_year", hue="is_canceled",data= df1,ec = "black")
plt.title("Checked-in vs Canceled Bookings by Arrival Year",fontsize=15,weight="bold")
plt.xlabel("Arrival Year",weight= "bold")
plt.ylabel("Booking Count",weight= "bold")
plt.legend(loc="upper right")
plt.grid(alpha = 0.5,axis = "y")


# In[56]:


# Concat the arrival year and month
df1["period"] = df1["arrival_date_year"].astype(str) +" "+ df1["arrival_date_month"]

# Plot Period vs Bookings
plt.subplots(figsize=(16,7))
sns.set_style("whitegrid")
sns.countplot(x="period", hue="is_canceled",data= df1,ec = "black")
plt.xticks(rotation=45)
plt.title("Checked-in vs Canceled Bookings over 3 year period",weight= "bold",fontsize= 15)
plt.xlabel("Months and years",weight="bold")
plt.ylabel("Total Bookings count",weight="bold")
plt.legend(loc="upper left")
plt.grid(alpha = 0.5,axis = "y")
plt.show() 


# In[57]:


## Which hotel type had more percentage of cancellations ## 

# Plot Month and Cancellations in respect to Hotel type
plt.subplots(figsize=(16,8))
sns.set_style("whitegrid")
sns.barplot(x="arrival_date_month", y="is_canceled",hue="hotel",data = df, order=months_order, palette="Set1",ec = "black")
plt.title("Percentage of Cancellations per month for each Hotel type", weight="bold", fontsize=15)
plt.xlabel("Month",weight="bold")
plt.ylabel("Cancellation Percentage",weight="bold")
plt.grid(alpha = 0.5,axis = "y")
plt.show()

The above Chart shows:
- City Hotels had Highest Cancellations in April month and Lowest in September.
- Resort Hotels had Highest Cancellations in August month, and Lowest in January.
# In[58]:


## What is ADR and its effect on booking and hotel type ## 
# Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights.

# Arrange the month for plotting
df1["arrival_date_month"] = pd.Categorical(df1["arrival_date_month"], categories=months_order, ordered=True)

# Plot Line Chart
plt.figure(figsize=(14,6))
sns.set_style("ticks")
sns.lineplot(x = "arrival_date_month", y = "adr", hue="is_canceled",data=df1)
plt.title("Bookings by Average Daily Rate (ADR) and Arrival Month", weight = "bold")
plt.xlabel("Arrival Month")
plt.ylabel("Average Daily Rate")
plt.xticks(rotation=45)
plt.legend(loc="upper right")
plt.grid(alpha = 0.5)

The above Chart shows:-
- ADR Average Daily Rate was more for customers who Checked-in (Obvious).
- ADR keeps on Raising in from June to August and drops back after that which means summer season had more demand in Hotel rooms
# In[59]:


## What is total booking count by each market segment customers?? ## 

# Extract Count of Booking types based on Market Segment
df1.groupby("market_segment")["is_canceled"].value_counts()


# In[60]:


plt.subplots(figsize=(12,6))
sns.countplot(x="market_segment", hue="is_canceled", data=df1,ec = "black")
plt.title("Hotel Booking status by Market Segment",weight= "bold")
plt.xlabel("Market Segment",weight = "bold")
plt.ylabel("Bookings count",weight = "bold")
plt.legend(loc="upper right")
plt.grid(alpha = 0.5,axis = "y")

The above Chart shows:- 
- Online Travel Agencies had more Checked-in bookings and Canceled bookings.
- Offline Travel Agencies, groups and direct market segments are major source of bookings preffered by customers.
# In[61]:


## What is booking types based on customer type. ## 
# Type of customer type 
df1["customer_type"].value_counts(normalize = True)*100


# In[62]:


# Types of customer types by booking status
df1.groupby("customer_type")["is_canceled"].value_counts()


# In[63]:


# Plot barchart
plt.subplots(figsize=(12,6))
sns.countplot(x="customer_type", hue="is_canceled", data=df1,ec = "black")
plt.title("Hotel Bookings by Customer type",weight= "bold", fontsize=15)
plt.xlabel("Market Segment",weight = "bold")
plt.ylabel("Bookings count",weight = "bold")
plt.legend(loc="upper right")
plt.grid(alpha = 0.5,axis = "y")

The above Chart shows:-
- Transient customers had the most bookings.
- Groups customers had lowest bookings.
# In[64]:


## What is total booking counts based on the Deposit type?? ## 

# plot barchart
plt.subplots(figsize=(12,6))
sns.countplot(x= "deposit_type",hue="is_canceled",data=df1,ec = "black")
plt.title("Hotel Bookings by Deposit Type", fontsize = 15,weight = "bold")
plt.xlabel("Deposit type",weight="bold")
plt.ylabel("Bookings Count",weight="bold")
plt.legend(loc="upper right")
plt.grid(alpha = 0.5,axis = "y")

The above Chart shows:-
- Most of the Cancelations where from no deposit customers.
- Intresting here is the Non refund type customers mostly checked-in.
# In[65]:


## What is the Total Bookings count by each Meal type opted by customers ? ## 
# Meal type and Customer count
df1.groupby("meal")["is_canceled"].value_counts()


# In[66]:


plt.subplots(figsize=(12,7))
sns.countplot(x="meal", hue="is_canceled",data=df1, ec = "black")
plt.title("Hotel Bookings by Meal type", fontsize = 15,weight = "bold")
plt.ylabel("Booking Count", weight="bold")
plt.xlabel("Meal Type", weight="bold")
plt.legend(loc="upper right")
plt.grid(alpha= 0.5, axis="y")


# In[67]:


## How many Bookings were done based on Guest's Freqency Type? ##

# Replace values for is_repeated guest Column for plotting
df1["is_repeated_guest"]= np.where(df1["is_repeated_guest"]==0,"Not_Repeated_Guest","Repeated_Guest")
df1["is_repeated_guest"].value_counts()


# In[68]:


plt.subplots(figsize=(12,6))
sns.countplot(x="is_canceled", hue="is_repeated_guest",data=df1,ec = "black", palette="Set2")
plt.title("Hotel Bookings by Repeated Guest", fontsize = 15,weight = "bold")
plt.ylabel("Booking Count", weight="bold")
plt.legend(loc="upper right")
plt.xlabel("Booking Status", weight="bold")
plt.grid(alpha = 0.5,axis = "y")


# In[69]:


## What is the Total Bookings count by Required car space opted by customers ? ## 
df1.groupby("required_car_parking_spaces")["is_canceled"].value_counts()


# In[70]:


plt.subplots(figsize=(12,6))
sns.countplot(x="required_car_parking_spaces", hue="is_canceled",data=df1,ec = "black")
plt.title("Hotel Bookings by Required car spaces", fontsize = 15,weight = "bold")
plt.ylabel("Booking Count", weight="bold")
plt.xlabel("Booking Status", weight="bold")
plt.legend(loc="upper right")
plt.grid(alpha = 0.5,axis = "y")


# In[71]:


## What is the Money spent by customers based Market type?? ##

plt.figure(figsize=(12,6))
sns.set_style("ticks")
ax = sns.barplot(x= "market_segment",y = "adr",data=df1,palette= "Paired",ec = "black")
ax.set_title("Average Daily Rate (ADR) by Customer's Market Type", fontsize = 14,weight = "bold")
ax.set_xlabel("Market Segment",weight = "bold")
ax.set_ylabel("Average Daily Rate (ADR)",weight = "bold")
plt.grid(alpha = 0.5,axis = "y")


# In[72]:


## Night spent by customer by hotel types ##
plt.subplots(figsize=(12,6))
sns.countplot(x="stays_in_weekend_nights", hue="hotel", data=df1, palette="Set1",ec="black")
plt.title("Nights spent by customers on Weekends",weight="bold")
plt.xlabel("Stay in Weekend Nights",weight="bold")
plt.ylabel("Bookings Count", weight="bold")
plt.legend(loc="upper right")
plt.xlim(-1,7)
plt.grid(alpha=0.5,axis="y")


# In[73]:


plt.subplots(figsize=(12,6))
sns.countplot(x="stays_in_week_nights", hue="hotel", data=df1, palette="Set1",ec="black")
plt.title("Nights spent by customers on Weekdays",weight="bold")
plt.xlabel("Stay in Weekend Nights",weight="bold")
plt.ylabel("Bookings Count", weight="bold")
plt.legend(loc="upper right")
plt.xlim(-1,11)
plt.grid(alpha=0.5,axis="y")


# In[74]:


## What are the total booking by each distribution channels?? ## 

# Count of ditribution channel"s bookings
df1.groupby(["distribution_channel"])["is_canceled"].value_counts()


# In[75]:


# Plot BarCharts
plt.subplots(figsize=(14,7))
sns.countplot(x="distribution_channel",data=df1, ec="black")
plt.title("Bookings by Booking Distribution Channels",weight="bold")
plt.xlabel("Distribution Channel",weight="bold")
plt.ylabel("Bookings Count", weight="bold")
plt.grid(alpha=0.5,axis="y")


# In[76]:


## Realtionship between lead time and booking status ##

# Convert lead time to days for plotting
df1["lead_time"]= df1["lead_time"]/24


# In[77]:


# plot BarChart
plt.subplots(figsize=(14,7))
sns.barplot(x="is_canceled",y="lead_time",data=df1,ec="black")
plt.title("Lead time in days vs Booking Status", weight="bold",fontsize=15)
plt.xlabel("Bookings Type",weight="bold")
plt.ylabel("Lead Days",weight="bold",fontsize=15)
plt.grid(alpha=0.5,axis="y")


# In[78]:


# plot BarChart
plt.subplots(figsize=(14,7))
sns.countplot(x="total_of_special_requests",hue="is_canceled", data=df1,ec="black")
plt.title("Total Number of Special Request", weight="bold",fontsize=15)
plt.xlabel("No. of Special requests",weight="bold")
plt.ylabel("Count",weight="bold",fontsize=15)
plt.grid(alpha=0.5,axis="y")


# In[79]:


plt.subplots(figsize=(12,6))
sns.set_style('ticks')
ax = sns.boxplot(x= 'deposit_type',y = 'adr',data=df,width = 0.7,linewidth= 2,palette= 'Paired')
ax.set_title("Effect of Average Daily Rate (ADR) by Deposit Type", fontsize = 14,weight = 'bold')
ax.set_xlabel("Market Segment",weight = 'bold')
ax.set_ylabel("Average Daily Rate (ADR)",weight = 'bold')
plt.grid(alpha = 0.5,axis = 'y')
ax.set(ylim=(0,500));


# In[80]:


plt.figure(figsize=(12,6))
sns.set_style('ticks')
ax = sns.violinplot(x= 'customer_type',y = 'adr',data=df,width = 0.7,linewidth= 2,palette= 'Set1')
ax.set_title("Effect of Average Daily Rate (ADR) by Customer Type", fontsize = 14,weight = 'bold')
ax.set_xlabel("Customer Type",weight = 'bold')
ax.set_ylabel("Average Daily Rate (ADR)",weight = 'bold')
plt.grid(alpha = 0.5,axis = 'y')
ax.set(ylim=(0,500));

The above Chart shows:-
- Transient customers spends more
- Contract customers spends very less compared to other customer type.
# In[81]:


df.shape


# # Data Cleaning and Data Transformation  
#    
#    Feature Extraction 
#    
#  - Making new column named Total guests which is total visitors count.
#  - By doing this we also reducing the number of attributes in a dataset while keeping as much of the variation in the original   dataset as possible.

# In[82]:


# Making single guests column which includes Children, Adult and Babies counts.

df["total_guests"]= df["children"]+df["adults"]+df["babies"]


# In[83]:


# dropping Adult, Children and Babies 
df.drop(["babies","adults","children"],axis=1,inplace=True)


# In[84]:


df.shape


# In[87]:


## Other Manipulations 
# Change the lead time from hours to days and rounding it off with 2 decimal points.
df["lead_time"] = (df["lead_time"]/24).round(2)


# In[88]:


## Manual encoding hotels types and month columns 
df['hotel'] = df['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
df['arrival_date_month'] = df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
df['reserved_room_type'] = df['reserved_room_type'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'L': 8})


# In[89]:


## Replace and Remove undefined features ## 

# replace meal Undefined with Self Catering 
df["meal"].replace("Undefined", "SC", inplace=True)
# Replace 
df["market_segment"].replace("Undefined", "Online TA", inplace=True)
df.drop(df[df['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)


# In[90]:


## Based on the In-Sights from EDA, We Drop Columns which are not much impacting the target variable ## 

df.drop(columns=['arrival_date_week_number',"reservation_status","reservation_status_date",
                 "assigned_room_type",'agent','required_car_parking_spaces', 'is_repeated_guest'], inplace=True, axis=1)


# In[91]:


## Types of Features ## 
# Numerical Features 
num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print('Num of Numerical Features :', len(num_features))


# In[92]:


# Categorical Features 
cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print('Num of Categorical Features :', len(cat_features))


# In[93]:


# Discrete Features 
discrete_feature=[feature for feature in num_features if len(df[feature].unique())<=25]
print('Num of Discrete Features :',len(discrete_feature))


# In[94]:


year_features=[feature for feature in df.columns if 'date' in feature or 'month' in feature]
print('Num of Year Features :',len(year_features))


# In[95]:


# Continous Features 
continuous_feature=[feature for feature in num_features if feature not in discrete_feature+year_features]
print('Num of Continuous Features :',len(continuous_feature))

continuous_feature = [feature for feature in num_features if feature not in discrete_feature + year_featurs]
print('num of continuous features:',)


# # Feature Selection
# 
# - Multicollinearity occurs when there are two or more independent variables in a multiple regression model, which have a high     correlation among themselves. When some features are highly correlated.
# - Multicollinearity can be detected using various techniques, one such technique being the Variance Inflation Factor(VIF).

# In[96]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def compute_vif(considered_features, df):
    
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


# In[97]:


compute_vif(num_features, df)


# In[ ]:


## Outlier Removal and Checking Skewness ## 

- For Skewed distributions: Use Inter-Quartile Range (IQR) proximity rule.

- The data points which fall below Q1 – 1.5 IQR or above Q3 + 1.5 IQR are outliers.

- where Q1 and Q3 are the 25th and 75th percentile of the dataset respectively, and IQR represents the inter-quartile range and given by Q3 – Q1.


# In[98]:


for i in continuous_feature:
        plt.figure(figsize=(10,6))
        sns.set_style('ticks')
        ax = sns.boxplot(df[i])


# In[99]:


def detect_outliers(col):
    # Finding the IQR
    percentile25 = df[col].quantile(0.25)
    percentile75 = df[col].quantile(0.75)
    print('\n ####', col , '####')
    print("percentile25",percentile25)
    print("percentile75",percentile75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    print("Upper limit",upper_limit)
    print("Lower limit",lower_limit)
    df.loc[(df[col]>upper_limit), col]= upper_limit
    df.loc[(df[col]<lower_limit), col]= lower_limit    
    return df


# In[100]:


for col in continuous_feature:
         detect_outliers(col)


# In[101]:


for i in continuous_feature:
        plt.figure(figsize=(10,6))
        sns.set_style('ticks')
        ax = sns.boxplot(df[i])


# In[102]:


df.drop(['previous_bookings_not_canceled', 'days_in_waiting_list'], inplace=True, axis=1)


# In[103]:


## Check Distribution of Lead Time column ## 
plt.figure(figsize=(10,6))
sns.set_style('ticks')
ax = sns.histplot(df['lead_time'],bins = 15,color = 'r',kde = True)
ax.set_title('Distribution of Lead Time - Right Skewed',fontsize = 14, weight='bold')
ax.set_xlabel("Lead Time",weight = 'bold')
ax.set_ylabel("Density",weight = 'bold');


# In[104]:


df[['lead_time']] = df[['lead_time']].apply(np.log1p)


# In[105]:


plt.figure(figsize=(10,6))
sns.set_style('ticks')
ax = sns.histplot(df['lead_time'],bins = 20,color = 'g', kde= True)
ax.set_title('Distribution of Lead Time - After Log Transformation',fontsize = 14, weight='bold')
ax.set_xlabel("Lead Time",weight = 'bold')
ax.set_ylabel("Density",weight = 'bold');


# In[106]:


df.dropna(inplace=True)


# # DataFrame Split
# - Split Dataframe to X and y
# - Here we set a variable X i.e, independent columns, and a variable y i.e, dependent column as the “is_canceled” column.

# In[107]:


from sklearn.model_selection import train_test_split
X = df.drop(['is_canceled'], axis=1)
y = df['is_canceled']


# In[108]:


## Save the list of countries to use it in the app ## 
import pickle
countries = list(X.country.unique())
with open('countryname.pkl', 'wb') as handle:
    pickle.dump(countries, handle)


# In[109]:


## Check all the columns which are train data set.
all_columns  = list(X.columns)
print(all_columns) 
len(all_columns)


# # Feature Encoding 
#  - Extracting Categorical features from train set for feature encoding.
#  

# In[110]:


cat_features = [feature for feature in X.columns if X[feature].dtype == 'O']
print('Num of Categorical Features :', len(cat_features))


# In[111]:


## Cheking for Unique varibles for each columns 
for feature in cat_features:
    print(feature,':', X[feature].nunique())


# # Pipeline for data Transformation 
# 
# - One Hot Encoding for Columns which had lesser unique values and not ordinal.
# - One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms   to do a better job in prediction.
# 
# -Binary Encoder is used for Country which had 170 unique values. 
# - Binary encoding is a combination of Hash encoding and one-hot encoding. In this encoding scheme, the categorical feature is     first converted into numerical using an ordinal encoder. Then the numbers are transformed in the binary number. After that       binary value is split into different columns.
# 
# - Binary encoding works really well when there are a high number of categories, Like Countries in our case.

# In[121]:


# Create Column Transformer with 3 types of transformers



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
category_encoders.binary.BinaryEncoder
num_features = [feature for feature in X.columns if X[feature].dtype != "O"]
oh_columns = ["meal", "market_segment", "distribution_channel", "deposit_type", "customer_type"]
bins_columns = ["country"]


numeric_transformer = StandardScaler()
bin_transfomer = BinaryEncoder()
oh_transfomer = OneHotEncoder()


preprocessor = ColumnTransformer(
[
    ('binary', bin_transformer, bin_columns),
    ('oh', oh_transformer, oh_columns),
    ('num', numeric_transformer, num_features)
]
)


# In[120]:


pip install --upgrade category_encoders


# In[122]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
category_encoders.binary.BinaryEncoder
num_features = [feature for feature in X.columns if X[feature].dtype != "O"]
oh_columns = ["meal", "market_segment", "distribution_channel", "deposit_type", "customer_type"]
bins_columns = ["country"]


numeric_transformer = StandardScaler()
bin_transfomer = BinaryEncoder()
oh_transfomer = OneHotEncoder()


preprocessor = ColumnTransformer(
[
    ('binary', bin_transformer, bin_columns),
    ('oh', oh_transformer, oh_columns),
    ('num', numeric_transformer, num_features)
]
)


# In[124]:


pip install sklearn


# In[131]:


pip install preprocessor


# In[132]:


pip install --upgrade preprocessor


# In[133]:


import preprocessor


# In[135]:


import pickle


# # Train Test Split

# In[138]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=36)
X_train.shape, X_test.shape


# In[ ]:





# In[140]:


pip install xgboost


# In[ ]:


conda install -c conda-forge xgboost


# In[ ]:


## Model Selection 
#- Here should understand the Various Classification models with default values from these models we can choose top 4 with Highest Accuracy score and proceed with HyperParameter Tuning

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay,                             precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[ ]:


models = {
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(),
     "K-Neighbors Classifier": KNeighborsClassifier(),
    "XGBClassifier": XGBClassifier(), 
     "CatBoosting Classifier": CatBoostClassifier(verbose=False),
     "Support Vector Classifier": SVC(),
    "AdaBoost Classifier": AdaBoostClassifier()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall


    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    
    print('='*35)
    print('\n')


# In[ ]:


## Hyper Parameter Tuning
knn_params = {"n_neighbors": [2, 3, 10, 20, 40, 50]}

rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}

cat_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [5, 8, 12, 20, 30]}


# In[ ]:


randomcv_models = [('KNN', KNeighborsClassifier(), knn_params),
                   ("RF", RandomForestClassifier(), rf_params),
                   ('XGBoost', XGBClassifier(), xgboost_params),
                   ('CatBoost', CatBoostClassifier(verbose=False), cat_params)
                   ]


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    rf_random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    rf_random.fit(X_train, y_train)
    model_param[name] = rf_random. best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


# In[ ]:


## Model Re-trained With Best Parameters.
from sklearn.metrics import roc_auc_score,roc_curve
models = {
    "Random Forest": RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features= 7, max_depth= None),
    "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=10),
    "XGBClassifier": XGBClassifier(n_estimators= 200, max_depth= 30, learning_rate= 0.1, colsample_bytree= 0.4, n_jobs=-1), 
    "CatBoosting Classifier": CatBoostClassifier(max_depth= 12, learning_rate= 0.1,verbose=False),
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc



    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))
    
    print('='*35)
    print('\n')


# In[ ]:


## Plot ROC-AUC Curve ## 

from sklearn.metrics import roc_auc_score,roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'Random Forest Classifier',
    'model': RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features= 'auto', max_depth= None),
    'auc': 0.8618
},
{
    'label': 'XGBoost Classifier',
    'model': XGBClassifier(n_estimators= 200, max_depth= 20, learning_rate= 0.1, colsample_bytree= 0.8, n_jobs=-1),
    'auc': 0.8073
},
{
    'label': 'KNN Classifier',
    'model': KNeighborsClassifier(n_neighbors=10),
    'auc': 0.8629 
},
{
    'label': 'CatBoost Classifier',
    'model': CatBoostClassifier(max_depth= 12, learning_rate= 0.1,verbose=False),
    'auc': 0.8615
},
    
]

# Below for loop iterates through your models list
for m in auc_models:
    model = m['model'] # select the model
    model.fit(X_train, y_train) # train the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], m['auc']))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positwive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(r"./images/roc_auc/auc.png")
plt.show()   # Display


# In[ ]:


## Feature selection for model deployement ## 
best_xgb = XGBClassifier(n_estimators= 200, max_depth= 30, learning_rate= 0.1, colsample_bytree= 0.4, n_jobs=-1)
best_xgb = best_xgb.fit(X_train,y_train)
xgb_pred = best_xgb.predict(X_test)
score = accuracy_score(y_test,xgb_pred)
cr = classification_report(y_test,xgb_pred)

print("FINAL XGB")
print ("Accuracy Score value: {:.4f}".format(score))
print (cr)


# In[ ]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(best_xgb, X_test, y_test)


# In[ ]:


import pickle
 
# Save the trained model as a pickle file.
pickle.dump(best_xgb, open('classificationmodel.pkl', 'wb'))


# In[ ]:




