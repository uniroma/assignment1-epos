# ASSIGNMENT 1

# First step: create the virtual environment with: 
            # python3 -m venv name_of_my_environment

# Second step: activate the virtual environment with: 
            # source name_of_my_environment/bin/activate

# Third step: install the pandas library with:
            # pip3 install pandas
            # pip3 install matplotlib


# Let's import three libraries:
import pandas as pd
from numpy.linalg import solve
import numpy as np

# Load the dataset:
df = pd.read_csv('~/Downloads/current.csv')
# Clean the DataFrame by removing the row with transformation codes:
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')

# Check df_cleaned containing the data cleaned:
df_cleaned

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# Function to apply transformations based on the transformation code:
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes:
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

# Since some transformations induce missing values, we drop the first two observations of the dataset:
df_cleaned = df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)

# Display the first few rows of the cleaned DataFrame:
df_cleaned.head()

# Let's import:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plot the transformed series:
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production',
                'Inflation (CPI)',
                '3-month Treasury Bill rate']


# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.show()


# FORECASTING WITH ARX MODEL

#1. Let's develop the model to forecast the INDPRO variable:
    # Extract the Target Variable (Yraw): select the column INDPRO from df_cleaned and assign it to Yraw
    # Extract the Explanatory Variables (Xraw): select the columns CPIAUCSL and TB3MS from df_cleaned and assign them to Xraw
Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

#Set the number of Lags (p) and Leads (h)
num_lags  = 4   
                # Four past observations as input to predict
num_leads = 1  
                # One-step ahead forecast. We are predicting the value one step into the future

# Create an empty DataFrame to store the predictor variables:
X = pd.DataFrame()

# Add the lagged values of Y (Target Variable) to capture autocorrelation (influence of past observations on future values):
col = 'INDPRO'
for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix:
        X[f'{col}_lag{lag}'] = Yraw.shift(lag)

# Perform the same operations as the first loop for each variable in Xraw:
for col in Xraw.columns:
    for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix:
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)

# Add a column of ones to the DataFrame X at position 0 (for the intercept):
X.insert(0, 'Ones', np.ones(len(X)))


# X is now a DataFrame:
X.head()
        # Note that the first p=4 rows of X have missing values

# The vector y can be similarly created as:
y = Yraw.shift(-num_leads)
y

# Note that: 
            # The variable y has missing values in the last h positions
            # We must keep the last row of the DataFrame X to build the model 

# Save the last row of X (converted to numpy to facilitate data processing):
X_T = X.iloc[-1:].values

# Subset to gey only rows of X and y from p+1 to h-1
# and convert to numpy array: 
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

# Let's check the values of X_T: 
X_T

# NOW WE CAN ESTIMATE THE PARAMETERS AND OBTAIN THE FORECAST:

# First import the function solve:
from numpy.linalg import solve

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X.T @ X, X.T @ y)

# Produce the One step ahead forecast:
# % change month-to-month INDPRO
forecast = X_T@beta_ols*100
forecast
            # The variable forecast now contains the one-step ahead of the variable INDPRO.
            # We are forecasting the percentage change because INDPRO has been transformed. 

# REAL-TIME EVALUATION: 

# We set the last observation at 12/1/1999 and start calculating the forecast:
def calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999',target = 'INDPRO', xvars = ['CPIAUCSL', 'TB3MS']):

    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]

    X = pd.DataFrame()
    for lag in range(0,p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)

    for col in Xraw.columns:
        for lag in range(0,p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
        if 'Ones' not in X.columns:
            X.insert(0, 'Ones', np.ones(len(X)))
    
    X_T = X.iloc[-1:].values
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        Yhat.append(X_T@beta_ols*100)
    return np.array(Y_actual) - np.array(Yhat)

# With this function, we calculate real-time errors by looping over the end date to ensure we end the loop at the right time.


t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = t0)
    e.append(ehat.flatten())
    T.append(t0)

## Create a pandas DataFrame from the list
edf = pd.DataFrame(e)
## Calculate the RMSFE, that is, the square root of the MSFE
np.sqrt(edf.apply(np.square).mean())

# Let's plot RMSFE for each 'h' value
# Data for the x-axis (h values)
h_values = [1, 4, 8]

# RMSFE values
rmsfe_values = np.sqrt(edf.apply(np.square).mean()) 

# Creating the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(h_values, rmsfe_values, marker='o', color='Red', linestyle='None')  # Plot the graph
plt.title('Root Mean Square Forecast Error (RMSFE) for Different Forecast Horizons (h)')  # Title of the graph
plt.xlabel('Forecast Horizon (h)')  # x-axis label
plt.ylabel('RMSFE')  # y-axis label
plt.grid(True)  # Show grid on the graph
plt.tight_layout()  # Set layout
plt.show()  # Show the graph
# The plot shows the RMSFE for each value of 'h'. In such a way we can see the accuracy of our model in the 1 month
# forecast, in the 4 and 8 month ones


#2.
# LET'S FORECAST CPIAUCSL (consumer price index) using:
    # Real Personal Income (RPI)
    # Unemployment Rate (UNRATE)
    # 3-Month Treasury Bill (TB3MS)
    # Personal Consumption Expenditure (PCEPI)

# The cleaned transformed Dataset is still:
df_cleaned

# Plot the transformed series:
series_to_plot2 = ['CPIAUCSL', 'RPI', 'UNRATE', 'TB3MS', 'PCEPI']
series_names2 = ['Inflation (CPI)', 
                 'Real Personal Income',
                 'Unemployment Rate', 
                 '3-Month Treasury Bill', 
                 'Personal Consumption Expenditure']

 # Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot2), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name2, plot_title in zip(axs, series_to_plot2, series_names2):
    if series_name2 in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name2], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.show()

# FORECASTING CONSUMER PRICE INDEX WITH ARX MODEL
# We can write matrix X2 in the following way
Y2raw = df_cleaned['CPIAUCSL']
X2raw = df_cleaned[['RPI','UNRATE','TB3MS', 'PCEPI']]

num_lags  = 4  ## this is p
num_leads = 1  ## this is h
X2 = pd.DataFrame() #this line creates an empty DataFrame

## Add the lagged values of Y2 at the dataframe X2
col2 = 'CPIAUCSL'
for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X2[f'{col2}_lag{lag}'] = Y2raw.shift(lag)

## Add the lagged values of 'CPIAUCSL' and 'TB3MS' at the dataframe X2
for col2 in X2raw.columns:
    for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X2[f'{col2}_lag{lag}'] = X2raw[col2].shift(lag)
## Add a column on ones (for the intercept)
X2.insert(0, 'Ones', np.ones(len(X2)))


## X2 is now a DataFrame
X2.head()

# Now we create also y2
y2 = Y2raw.shift(-num_leads)
y2

# Now we create two numpy arrays with the missing values stripped:
# Save last row of X2 (converted to numpy)
X2_T = X2.iloc[-1:].values
## Subset getting only rows of X2 and y2 from p+1 to h-1
## and convert to numpy array
y2 = y2.iloc[num_lags:-num_leads].values
X2 = X2.iloc[num_lags:-num_leads].values

X2_T 

# NOW WE HAVE TO ESTIMATE THE PRAMETERS AND OBTAIN THE FORECAST

# Solving for the OLS estimator beta: (X2'X2)^{-1} X2'Y
beta_ols2 = solve(X2.T @ X2, X2.T @ y2)
forecast2 = X2_T@beta_ols2*100
forecast2

#REAL TIME EVALUATION
#### LET'S DO THIS for h= 1,4,8 ####
def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='12/1/1999', target='CPIAUCSL', xvars=['RPI','UNRATE','TB3MS', 'PCEPI']):

    rt_df2 = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    Y2_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y2_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target] * 100)
    Y2raw = rt_df2[target]
    X2raw = rt_df2[xvars]

    X2 = pd.DataFrame()
    for lag in range(0, p):
        X2[f'{target}_lag{lag}'] = Y2raw.shift(lag)

    for col2 in X2raw.columns:
        for lag in range(0, p):
            X2[f'{col2}_lag{lag}'] = X2raw[col2].shift(lag)

    if 'Ones' not in X2.columns:
        X2.insert(0, 'Ones', np.ones(len(X2)))

    X2_T = X2.iloc[-1:].values
    Y2hat = []
    for h in H:
        y2_h = Y2raw.shift(-h)
        y2 = y2_h.iloc[p:-h].values
        X2_ = X2.iloc[p:-h].values
        beta_ols2 = solve(X2_.T @ X2_, X2_.T @ y2)
        Y2hat.append(X2_T@beta_ols2*100)

    # Return Y2_actual, Y2hat and errors (e2hat)
    return np.array(Y2_actual), np.array(Y2hat), np.array(Y2_actual) - np.array(Y2hat)


# With this function, you can calculate real-time errors by looping over 
# the 'end_date' to ensure you end the loop at the right time.


t0 = pd.Timestamp('12/1/1999')
e2 = []
T = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    Y2_actual, Y2hat, e2hat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)
    e2.append(e2hat.flatten())
    T.append(t0)

#Print these values
print(f'Y_actual: {Y2_actual}')
print(f'Yhat: {Y2hat}')
print(f'ehat: {e2hat}')

## Create a pandas DataFrame from the list
edf2 = pd.DataFrame(e2)
## Calculate the RMSFE, that is, the square root of the MSFE
np.sqrt(edf2.apply(np.square).mean())

# Let's plot RMSFE for each 'h' value
# Data for the x-axis (h values)
h_values2 = [1, 4, 8]

# RMSFE values
rmsfe_values2 = np.sqrt(edf2.apply(np.square).mean()) 

# Creating the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(h_values2, rmsfe_values2, marker='o', color='Red', linestyle='None')  # Plot the graph
plt.title('Root Mean Square Forecast Error (RMSFE) for Different Forecast Horizons (h)')  # Title of the graph
plt.xlabel('Forecast Horizon (h)')  # x-axis label
plt.ylabel('RMSFE')  # y-axis label
plt.grid(True)  # Show grid on the graph
plt.tight_layout()  # Set layout
plt.show()  # Show the graph

# Let's try to forecast Real Personal Income (RPI) using:
# CPI (CPIAUCSL)
# Unemployment Rate (UNRATE)
# 3-Month Treasury Bill (TB3MS)
# Personal Consumption Expenditure (PCEPI)
# Real Money Stock (M2REAL)

# The cleaned transformed dataset is, as always
df_cleaned

## Plot the transformed series
series_to_plot3 = ['CPIAUCSL', 'RPI', 'UNRATE', 'TB3MS', 'PCEPI', 'M2REAL']
series_names3 = ['Inflation (CPI)','Real Personal Income', 
                 'Unemployment Rate', '3-Month Treasury Bill', 
                 'Personal Consumption Expenditure', 'Real Money Stock']
fig, axs = plt.subplots(len(series_to_plot3), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one
for ax, series_names3, plot_title in zip(axs, series_to_plot3, series_names3):
    if series_names3 in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_names3], label=plot_title)
        # Formatting the x-axis to show only every five years
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        # Improve layout of date labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available
plt.tight_layout()
plt.show()


#LET'S FORECAST
Y3raw = df_cleaned['RPI']
X3raw = df_cleaned[['CPIAUCSL', 'TB3MS', 'UNRATE','PCEPI', 'M2REAL']]

#Set the number of Lags (p) and Leads (h)
num_lags  = 4   
num_leads = 1  

X3 = pd.DataFrame()

col3 = 'RPI'
for lag in range(0,num_lags+1):
    X3[f'{col3}_lag{lag}'] = Y3raw.shift(lag)


for col3 in X3raw.columns:
    for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix:
        X3[f'{col3}_lag{lag}'] = X3raw[col3].shift(lag)
        
X3.insert(0, 'Ones', np.ones(len(X3)))

X3.head()

# The vector y can be similarly created as:
y3 = Yraw.shift(-num_leads)
y3

X3_T = X3.iloc[-1:].values

# Subset to gey only rows of X and y from p+1 to h-1
# and convert to numpy array: 
y3 = y3.iloc[num_lags:-num_leads].values
X3 = X3.iloc[num_lags:-num_leads].values

X3_T

beta_ols3 = solve(X3.T @ X3, X3.T @ y3)
forecast3 = X3_T@beta_ols3*100
forecast3

def calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999',target = 'RPI', xvars = ['CPIAUCSL', 'TB3MS', 'UNRATE','PCEPI']):

    rt_df3 = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    Y3_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y3_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)

    Y3raw = rt_df3[target]
    X3raw = rt_df3[xvars]

    X3 = pd.DataFrame()
    for lag in range(0,p):
        X3[f'{target}_lag{lag}'] = Y3raw.shift(lag)

    for col3 in X3raw.columns:
        for lag in range(0,p):
            X3[f'{col3}_lag{lag}'] = X3raw[col3].shift(lag)
        if 'Ones' not in X3.columns:
            X3.insert(0, 'Ones', np.ones(len(X3)))
    
    X3_T = X3.iloc[-1:].values
    Yhat3 = []
    for h in H:
        y3_h = Y3raw.shift(-h)
        y3 = y3_h.iloc[p:-h].values
        X3_ = X3.iloc[p:-h].values
        beta_ols3 = solve(X3_.T @ X3_, X3_.T @ y3)
        Yhat3.append(X3_T@beta_ols3*100)
    return np.array(Y3_actual), np.array(Yhat3), np.array(Y3_actual) - np.array(Yhat3)
t0 = pd.Timestamp('12/1/1999')
e3 = []
T = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    Y3_actual, Yhat3, e3hat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)
    e3.append(e3hat.flatten())
    T.append(t0)

#Print these values
print(f'Y_actual: {Y3_actual}')
print(f'Yhat: {Yhat3}')
print(f'ehat: {e3hat}')

## Create a pandas DataFrame from the list
edf3 = pd.DataFrame(e3)
## Calculate the RMSFE, that is, the square root of the MSFE
np.sqrt(edf3.apply(np.square).mean())

# Let's plot RMSFE for each 'h' value
# Data for the x-axis (h values)
h_values3 = [1, 4, 8]

# RMSFE values
rmsfe_values3 = np.sqrt(edf3.apply(np.square).mean()) 

# Creating the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(h_values3, rmsfe_values3, marker='o', color='Red', linestyle='None')  # Plot the graph
plt.title('Root Mean Square Forecast Error (RMSFE) for Different Forecast Horizons (h)')  # Title of the graph
plt.xlabel('Forecast Horizon (h)')  # x-axis label
plt.ylabel('RMSFE')  # y-axis label
plt.grid(True)  # Show grid on the graph
plt.tight_layout()  # Set layout
plt.show()  # Show the graph