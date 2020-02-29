# Import pandas, numpy, matplotlib, and seaborn.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# The data shows almost no obvious correlation,
# most people in this dataset spend an average amount of time and an average amount of money
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)


# Do the same but with the Time on App column instead.
# this jointplot shows a clear positive correlation between the time on app and money spent
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)


# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x="Time on App", y="Length of Membership", data=customers, kind='hex')


# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)

#????? what plot

# Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?

# NULL

# Create a linear model plot(using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
p = sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers)
#plt.show()

# customers and a variable y equal to the "Yearly Amount Spent" column.
# ?

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101

# to use model_selection I have to import it from sklearn first
from sklearn.model_selection import train_test_split
# the instructions are pretty vague so I'll just assume we are trying to find what makes ammount spent go up
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# Train/fit lm on the training data.
lm.fit(X_train, y_train)

# Print out the coefficients of the model
print(pd.DataFrame(lm.coef_, X.columns, columns=['Coefficents']))

# Use lm.predict() to predict off the X_test set of the data.


# Create a scatterplot of the real test values versus the predicted values.


# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas


# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().


# Recreate the dataframe below.


# How can you interpret these coefficients?


# Do you think the company should focus more on their mobile app or on their website?


# Ref:www.pieriandata.com
