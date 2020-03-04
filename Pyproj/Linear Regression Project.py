# Name: Alec Dosier
# NetID: ald170830
# Date: 3/1/2020
# Class: 4375.502 Introduction to Machine Learning
# -------------------------------------------------

# Import pandas, numpy, matplotlib, and seaborn.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
# --------------- ENVIRONMENT CONFIGUREATION
# This allows functions like head() to display more than a single column in the console
pd.set_option('display.max_columns', None)

# Read in the Ecommerce Customers csv file as a DataFrame called customers.
customers = pd.read_csv("Ecommerce Customers")

# Check the head of customers, and check out its info() and describe() methods.
print('Head -----------------------')
print(customers.head())
print('Info -----------------------')
customers.info()
print('Describe -------------------')
print(customers.describe())

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?

#   The data shows almost no obvious correlation,
#   most people in this dataset spend an average amount of time and an average amount of money
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)

# Do the same but with the Time on App column instead.
# this jointplot shows a clear positive correlation between the time on app and money spent
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)


# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x="Time on App", y="Length of Membership", data=customers, kind='hex')


# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.
# (Don't worry about the the colors)
sns.pairplot(customers)

# Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
#   length of membership

# Create a linear model plot(using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
p = sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers)

# customers and a variable y equal to the "Yearly Amount Spent" column.
# Now that we've explored the data a bit, let's go ahead and split the data into training and
# testing sets. ** Set a variable X equal to the numerical features of the customers and
# a variable y equal to the "Yearly Amount Spent" column.

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets.
# Set test_size=0.3 and random_state=101

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# Train/fit lm on the training data.
lm.fit(X_train, y_train)

# Print out the coefficients of the model
print("MODEL COEFFICENTS =======================")
#   I assigned this to a df because it is the answer to the "recreate this dataframe" question later
modcoef = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficents'])
print(modcoef)

# Use lm.predict() to predict off the X_test set of the data.
prediction = lm.predict(X_test)

# Create a scatterplot of the real test values versus the predicted values.
plt.scatter(y_test, prediction)
plt.show()

# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas
MAE = metrics.mean_absolute_error(y_test, prediction)  # this finds the average error between predicted an actual
MSE = metrics.mean_squared_error(y_test, prediction)
RMSE = np.sqrt(MSE)
print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)

# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.distplot((y_test-prediction), bins=50)
plt.show()
# Recreate the dataframe below.
# ================================
#                       Coefficent|
# Avg. Session Length   25.981550 |
# Time on App           38.590159 |
# Time on Website        0.190405 |
# Length of Membership  61.279097 |
# ================================

#   This dataframe is already exists in "modcoef"
print("recreation of the dataframe given to us ------------------")
print(modcoef)

# How can you interpret these coefficients?
#   These coefficients show how much the Yearly Ammount Spent is adjusted for every 1 unit that a feature increases
#   They also are an indicator of correlation between the target and the feature
# ================================
#                       Coefficent|
# Avg. Session Length   25.981550 |
# Time on App           38.590159 |
# Time on Website        0.190405 | * This shows no significant correlation
# Length of Membership  61.279097 |
# ================================

# Do you think the company should focus more on their mobile app or on their website?
#   This question depends on what the context of the word "focus" is.
#   Based on the data above, their revenue comes from their time spent on the app.
#       You could say that they should focus on their app because it brings money in.
#   However, their Website is worthless from a revenue perspective.
#       If their website is meant to make sales, they need to focus on that so they can have
#       two sources of revenue. They could make their app perform worse if they keep adding superficial features to it.

#   The answer is based on the context of the business and not a choice that should be made purely from the data.

# Ref:www.pieriandata.com
