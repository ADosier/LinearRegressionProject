Import pandas, numpy, matplotlib, and seaborn.
Read in the Ecommerce Customers csv file as a DataFrame called customers.
Check the head of customers, and check out its info() and describe() methods.
Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
Do the same but with the Time on App column instead.
Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)
Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
Create a linear model plot(using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
customers and a variable y equal to the "Yearly Amount Spent" column.
Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101
Import LinearRegression from sklearn.linear_model
Create an instance of a LinearRegression() model named lm.
Train/fit lm on the training data.
Print out the coefficients of the model
Use lm.predict() to predict off the X_test set of the data.
Create a scatterplot of the real test values versus the predicted values.
Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas
Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
Recreate the dataframe below.
How can you interpret these coefficients?
Do you think the company should focus more on their mobile app or on their website?
# Ref: www.pieriandata.com
