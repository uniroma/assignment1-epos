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
df = pd.read_csv('/Users/valentinasanna/Desktop/magistrale/Primo anno /Secondo semestre/Comp Tools/Assignment_1/current.csv')

#We can check if the dataset has been loaded correctly executing:
df 

# Clean the DataFrame by removing the row with transformation codes:
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

# Check df_cleaned containing the data cleaned:
df_cleaned

# Extract transformation codes:
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# View transformation codes: 
transformation_codes

## Transformation_codes contains the transformation codes and their meaning:
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$

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

# Display the first few rows of the cleaned DataFrame:
df_cleaned.head()

## Plot the transformed series:
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production', 'Inflation (CPI)', 'Federal Funds Rate']
    # 'INDPRO'   for Industrial Production, 
    # 'CPIAUCSL' for Inflation (Consumer Price Index), 
    # 'TB3MS'    3-month treasury bill.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create a figure and a grid of subplots:
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one:
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
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



# LET'S DEVELOP THE MODEL:

# Define the variable Y by selecting the Industrial Production time series and removing all the missing values
# Define the variables X by selecting the Consumer Price Index and the Federal Funds Rate (utilized as predictors)
Y = df_cleaned['INDPRO'].dropna()
X = df_cleaned[['CPIAUCSL', 'FEDFUNDS']].dropna()

h = 1 
p = 4 
r = 4 
# Where: 
        # h: one-step ahead forecast. We are predicting the value one step into the future
        # p: order of autoregression. We are considering the four most recent observation of Y   
        # r: order of exogenous inputs. We are including the four most recent observations of X.

# Define the target variable Y: 
Y_target = Y.shift(-h).dropna()

# Y_lagged: variable Y lagged by i periods (where i goes from 0 to p), used as input in the model (indicates autocorrelation):
Y_lagged = pd.concat([Y.shift(i) for i in range(p+1)], axis=1).dropna()

# X_lagged: lagged versions of the exogenous variables, used as inputs in the prediction model:
X_lagged = pd.concat([X.shift(i) for i in range(r+1)], axis=1).dropna()

# Create an index representing the set of common rows between the lagged time series of Y and X
    # Each observation in the dataset has corresponding values for both predictors at the same time point:
common_index = Y_lagged.index.intersection(Y_target.index)
common_index = common_index.intersection(X_lagged.index)

# This is the last row needed to create the forecast:
X_T = np.concatenate([[1], Y_lagged.iloc[-1], X_lagged.iloc[-1]])

# Now we have X_T which contains all the necessary information to make a forecast with the ARMAX model. 
# It includes the most recent past values of the dependent variable Y, the most recent past values of 
# the exogenous variables X, and the constant term [1] (intercept) of the model.

# Next Step: we want to keep just the values of 'Y_lagged', 'Y_target' and 'X_lagged'
# By keeping only the rows that correspond to the same dates present in the common_index:
Y_target = Y_target.loc[common_index]
Y_lagged = Y_lagged.loc[common_index]
X_lagged = X_lagged.loc[common_index]

# In this way all three dataframes have consistent indices

# Here, we want to merge the 'X_lagged' and 'Y_lagged' DataFrames so that the columns of 'Y_lagged' are added 
# to the right of the columns of 'X_lagged'.
# The resulting dataframe X_reg contains all the variables (lagged values of exogenous and endogenous variables) 
# necessary for building our model.
X_reg = pd.concat([X_lagged, Y_lagged], axis = 1)

# Prepare the data for fitting the regression model:
X_reg_np = np.concatenate([np.ones((X_reg.shape[0], 1)), X_reg.values], axis=1)
Y_target_np = Y_target.values

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X_reg_np.T @ X_reg_np, X_reg_np.T @ Y_target_np)

print("Coefficients:")
for i, coef in enumerate(beta_ols):
    print(f"Beta_{i}: {coef}")

# Produce the One step ahead forecast
# % change month-to-month INDPRO
print(X_T)
forecast = X_T@beta_ols*100
print(forecast)
print(beta_ols)

#with this model we obtain both positive and negative coefficient and they are significant. 
# This means that the Consumer Price Index and the Federal Funds Rate have significant impacts on the Industrial Production Index.

#We can forecast inflation with the same dataset

I = df_cleaned['CPIAUCSL'].dropna()
Z = df_cleaned[['INDPRO', 'FEDFUNDS']].dropna()
#I=Y
h = 1 ## One-step ahead
p = 4
r = 4

I_target = I.shift(-h).dropna()
I_lagged = pd.concat([I.shift(i) for i in range(p+1)], axis=1).dropna()
Z_lagged = pd.concat([Z.shift(i) for i in range(r+1)], axis=1).dropna()
common_index = I_lagged.index.intersection(I_target.index)
common_index = common_index.intersection(Z_lagged.index)

## This is the last row needed to create the forecast
Z_T = np.concatenate([[1], I_lagged.iloc[-1], Z_lagged.iloc[-1]])

## Align the data
I_target = I_target.loc[common_index]
I_lagged = I_lagged.loc[common_index]
Z_lagged = Z_lagged.loc[common_index]

Z_reg = pd.concat([Z_lagged, I_lagged], axis = 1)



Z_reg = pd.concat([Z_lagged, I_lagged], axis=1)
Z_reg_np = np.concatenate([np.ones((Z_reg.shape[0], 1)), Z_reg.values], axis=1)
I_target_np = I_target.values


# It constructs the regression matrix X_reg by concatenating lagged predictor variables (X_lagged) and lagged target variables (Y_lagged).
# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols_i = solve(Z_reg_np.T @ Z_reg_np, Z_reg_np.T @ I_target_np)

## Produce the One step ahead forecast
print(Z_T)
forecast = Z_T@beta_ols_i*100
print(forecast)
print(beta_ols_i)

#also in this regression we have both positive and negative coefficients, but with lower values. In particolar Industrial Production Index
#have higher coefficients, suggesting a larger impact on Consumer Price Index than Federal Funds Rate.

# Let's try to forecast CPI (Inflation) using:
# Real Personal Income (RPI)
# Unemployment Rate (UNRATE)
# 3-Month Treasury Bill (TB3MS)
# Personal Consumption Expenditure (PCEPI)

# The cleaned transformed dataset is still
df_cleaned

## Plot the transformed series
series_to_plot2 = ['CPIAUCSL', 'RPI', 'UNRATE', 'TB3MS', 'PCEPI']
series_names2 = ['Inflation (CPI)','Real Personal Income', 'Unemployment Rate', '3-Month Treasury Bill', 'Personal Consumption Expenditure']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot2), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one
for ax, series_name2, plot_title in zip(axs, series_to_plot2, series_names2):
    if series_name2 in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name2], label=plot_title)
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

# LET'S BUILD THE MODEL

# Extract the series of data for the variable 'Inflation (CPI)' from the cleaned DataFrame
# and removing any rows with missing values, resulting in a new series 'Y'.
Y2 = df_cleaned['CPIAUCSL'].dropna()

# Extracting the series of data for variables 'RPI', 'UNRATE', 'TB3MS', 'PCEPI' 
# from the cleaned DataFrame and removing any rows with missing values, 
# resulting in a new DataFrame 'X'.
X2 = df_cleaned[['RPI', 'UNRATE', 'TB3MS', 'PCEPI']].dropna()

# Define indexes for our model
h = 1 ## One-step ahead
p = 4 ## Lags of Y2
r = 4 ## Lags of X2

# Define the target variable Y2: 
Y2_target = Y2.shift(-h).dropna()

# Y2_lagged: variable Y2 lagged by i periods (where i goes from 0 to p), used as input in the model (indicates autocorrelation):
Y2_lagged = pd.concat([Y2.shift(i) for i in range(p+1)], axis=1).dropna()

# X2_lagged: lagged versions of the exogenous variables, used as inputs in the prediction model:
X2_lagged = pd.concat([X2.shift(i) for i in range(r+1)], axis=1).dropna()

# Create an index representing the set of common rows between the lagged time series of Y2 and X2
    # Each observation in the dataset has corresponding values for both predictors at the same time point:
common_index2 = Y2_lagged.index.intersection(Y2_target.index)
common_index2 = common_index2.intersection(X2_lagged.index)

# This is the last row needed to create the forecast:
X2_T = np.concatenate([[1], Y2_lagged.iloc[-1], X2_lagged.iloc[-1]])

# Now we have X2_T which contains all the necessary information to make a forecast with the ARX model. 
# It includes the most recent past values of the dependent variable Y2, the most recent past values of 
# the exogenous variables X2, and the constant term [1] (intercept) of the model.

# Next Step: we want to keep just the values of 'Y2_lagged', 'Y2_target' and 'X2_lagged'
# By keeping only the rows that correspond to the same dates present in the common_index:
Y2_target = Y2_target.loc[common_index2]
Y2_lagged = Y2_lagged.loc[common_index2]
X2_lagged = X2_lagged.loc[common_index2]

# In this way all three Dataframes have consistent indices

# Here, we want to merge the 'X2_lagged' and 'Y2_lagged' DataFrames so that the columns of 'Y2_lagged' are added 
# to the right of the columns of 'X2_lagged'.
# The resulting dataframe X2_reg contains all the variables (lagged values of exogenous and endogenous variables) 
# necessary for building our model.
X2_reg = pd.concat([X2_lagged, Y2_lagged], axis = 1)

# Prepare the data for fitting the regression model:
X2_reg_np = np.concatenate([np.ones((X2_reg.shape[0], 1)), X2_reg.values], axis=1)
Y2_target_np = Y2_target.values

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols2 = solve(X2_reg_np.T @ X2_reg_np, X2_reg_np.T @ Y2_target_np)

# Produce the One step ahead forecast
# % change month-to-month INDPRO
print(X2_T)
forecast2 = X2_T@beta_ols2*100
print(forecast2)
print(beta_ols2)


# the secon set has larger magnitudes, and this could suggest a stronger influences of the predictors on the target variable compared 
# to the set runned before (with Z_T and beta_ols_i).
# The forecasted percentage change in the second set is higher, indicating a more pronounced 
# expected change in the target variable compared to the first set.

# Let's try to forecast Real Personal Income (RPI) using:
# CPI (CPIAUCSL)
# Unemployment Rate (UNRATE)
# 3-Month Treasury Bill (TB3MS)
# Personal Consumption Expenditure (PCEPI)

# The cleaned transformed dataset is, as always
df_cleaned

## Plot the transformed series
series_to_plot3 = ['CPIAUCSL', 'RPI', 'UNRATE', 'TB3MS', 'PCEPI']
series_names3 = ['Inflation (CPI)','Real Personal Income', 'Unemployment Rate', '3-Month Treasury Bill', 'Personal Consumption Expenditure']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot3), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one
for ax, series_names3, plot_title in zip(axs, series_to_plot3, series_names3):
    if series_names3 in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name2], label=plot_title)
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

# LET'S BUILD THE MODEL

# Extract the series of data for the variable 'Real Personal Income (RPI)' from the cleaned DataFrame
# and removing any rows with missing values, resulting in a new series 'Y'.
Y3 = df_cleaned['RPI'].dropna()

# Extracting the series of data for variables 'CPI', 'UNRATE', 'TB3MS', 'PCEPI' 
# from the cleaned DataFrame and removing any rows with missing values, 
# resulting in a new DataFrame 'X'.
X3 = df_cleaned[['CPIAUCSL', 'UNRATE', 'TB3MS', 'PCEPI']].dropna()

# Define indexes for our model
h = 1 ## One-step ahead
p = 4 ## Lags of Y3
r = 4 ## Lags of X3

# Define the target variable Y3: 
Y3_target = Y3.shift(-h).dropna()

# Y3_lagged: variable Y3 lagged by i periods (where i goes from 0 to p), used as input in the model (indicates autocorrelation):
Y3_lagged = pd.concat([Y3.shift(i) for i in range(p+1)], axis=1).dropna()

# X3_lagged: lagged versions of the exogenous variables, used as inputs in the prediction model:
X3_lagged = pd.concat([X3.shift(i) for i in range(r+1)], axis=1).dropna()

# Create an index representing the set of common rows between the lagged time series of Y3 and X3
    # Each observation in the dataset has corresponding values for both predictors at the same time point:
common_index3 = Y3_lagged.index.intersection(Y3_target.index)
common_index3 = common_index3.intersection(X3_lagged.index)

# This is the last row needed to create the forecast:
X3_T = np.concatenate([[1], Y3_lagged.iloc[-1], X3_lagged.iloc[-1]])

# Now we have X3_T which contains all the necessary information to make a forecast with the ARX model. 
# It includes the most recent past values of the dependent variable Y3, the most recent past values of 
# the exogenous variables X3, and the constant term [1] (intercept) of the model.

# Next Step: we want to keep just the values of 'Y3_lagged', 'Y3_target' and 'X3_lagged'
# By keeping only the rows that correspond to the same dates present in the common_index:
Y3_target = Y3_target.loc[common_index3]
Y3_lagged = Y3_lagged.loc[common_index3]
X3_lagged = X3_lagged.loc[common_index3]

# In this way all three Dataframes have consistent indices

# Here, we want to merge the 'X3_lagged' and 'Y3_lagged' DataFrames so that the columns of 'Y3_lagged' are added 
# to the right of the columns of 'X3_lagged'.
# The resulting dataframe X3_reg contains all the variables (lagged values of exogenous and endogenous variables) 
# necessary for building our model.
X3_reg = pd.concat([X3_lagged, Y3_lagged], axis = 1)

# Prepare the data for fitting the regression model:
X3_reg_np = np.concatenate([np.ones((X3_reg.shape[0], 1)), X3_reg.values], axis=1)
Y3_target_np = Y3_target.values

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols3 = solve(X3_reg_np.T @ X3_reg_np, X3_reg_np.T @ Y3_target_np)

# Produce the One step ahead forecast
# % change month-to-month RPI
print(X3_T)
forecast3 = X3_T@beta_ols3*100
print(forecast3)
print(beta_ols3)

#The negative coefficient and the negative forecast suggest that,
# there is an expectation for a decrease in the Real Personal Income in the next period.
#The coefficients are still all significant, even though vary on magnitudes of influence.

#LET'S FORECAST 3-Month Treasury Bill (TB3MS) USING: 
#CONSUMER PRICE INDICATOR (CPIAUCSL)
#Real Personal Income (RPI)
#Consumer Sentiment Index (UMCSENTx)
#PRODUCER PRICE INDEX (WPSFD49207)

df_cleaned.head()

## Plot the transformed series:
series_to_plot4 = ['TB3MS', 'CPIAUCSL', 'RPI', 'UMCSENTx', 'WPSFD49207']
series_names4 = ['Federal Funds Rate', 'Consumer Price indicator', 'Real Personal Income', 'Consumer Sentiment Index', 'Producer Price Index']

# Create a figure and a grid of subplots:
fig, axs = plt.subplots(len(series_to_plot4), 1, figsize=(10, 15))

# Iterate over the selected series and plot each one:
for ax, series_name4, plot_title in zip(axs, series_to_plot4, series_names4):
    if series_name4 in df_cleaned.columns:
        # Convert 'sasdate' to datetime format for plotting
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name4], label=plot_title)
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


# LET'S DEVELOP THE MODEL:
YT = df_cleaned['TB3MS'].dropna()
XT = df_cleaned[['CPIAUCSL', 'RPI', 'UMCSENTx', 'WPSFD49207']].dropna()

h = 1 
p = 4 
r = 4 

# Define the target variable Y: 
YT_target = Y.shift(-h).dropna()

# Y_lagged: variable Y lagged by i periods (where i goes from 0 to p), used as input in the model (indicates autocorrelation):
YT_lagged = pd.concat([Y.shift(i) for i in range(p+1)], axis=1).dropna()

# X_lagged: lagged versions of the exogenous variables, used as inputs in the prediction model:
XT_lagged = pd.concat([X.shift(i) for i in range(r+1)], axis=1).dropna()

# Create an index representing the set of common rows between the lagged time series of Y and X
    # Each observation in the dataset has corresponding values for both predictors at the same time point:
common_index4 = YT_lagged.index.intersection(YT_target.index)
common_index4 = common_index4.intersection(XT_lagged.index)

# This is the last row needed to create the forecast:
XT_T = np.concatenate([[1], YT_lagged.iloc[-1], XT_lagged.iloc[-1]])


# Next Step: we want to keep just the values of 'Y_lagged', 'Y_target' and 'X_lagged'
# By keeping only the rows that correspond to the same dates present in the common_index:
YT_target = YT_target.loc[common_index4]
YT_lagged = YT_lagged.loc[common_index4]
XT_lagged = XT_lagged.loc[common_index4]

# In this way all three dataframes have consistent indices

# The resulting dataframe X_reg contains all the variables (lagged values of exogenous and endogenous variables) 

XT_reg = pd.concat([XT_lagged, YT_lagged], axis = 1)

# Prepare the data for fitting the regression model:
XT_reg_np = np.concatenate([np.ones((XT_reg.shape[0], 1)), XT_reg.values], axis=1)
YT_target_np = YT_target.values

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_olsT = solve(XT_reg_np.T @ XT_reg_np, XT_reg_np.T @ YT_target_np)


# Produce the One step ahead forecast
print(XT_T)
forecast = XT_T@beta_olsT*100
print(forecast)
print(beta_olsT)

#The coefficients indicate that the variables have significant impacts on 3-Month Treasury Bill, 
# as evidenced by their relatively large magnitudes.
#The forecasted percentage change indicates a moderate expected change in 3-Month Treasury Bill for the next period.