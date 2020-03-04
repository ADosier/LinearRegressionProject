# ===============================
# Name: Alec Dosier
# NetID: ald170830
# Date: 3/1/2020
# Class: 4375.502 Intro To Machine Learning
# ===============================

###
# Linear Regression Project
# 1.  Read in bikeshare.csv file and set it to a dataframe called bike. 
bike = read.csv('bikeshare.csv')

### Bikshare Data Set Information
# . datetime - hourly date + timestamp 
# . season - 1 = spring, 2 = summer, 3 = fall, 4 = winter 
# . holiday - whether the day is considered a holiday
# . workingday - whether the day is neither a weekend nor holiday
# . weather - 
#   ???1: Clear, Few clouds, Partly cloudy, Partly cloudy 
#   ???2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
#   ???3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
#   ???4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 

# . temp - temperature in Celsius
# . atemp - "feels like" temperature in Celsius
# . humidity - relative humidity
# . windspeed - wind speed
# . casual - number of non-registered user rentals initiated
# . registered - number of registered user rentals initiated
# . count - number of total rentals
###

# 2.  Check the head of df

  # I am guessing we need to predict the number of total rentals (count)
  # because if someone hires a data scientitst, they want you to make them more money from data.

# 3.  Can you figure out what is the target we are trying to predict? 
#     Check the Kaggle Link above if you are confused on this. 

head(bike)


### 
# Exploratory Data Analysis

# 4.  Create a scatter plot of count vs temp. Set a good alpha value.
library(ggplot2)
qplot(temp, count, data=bike, alpha=0.1)

# 5.  Plot count versus datetime as a scatterplot with a color gradient 
#     based on temperature. You'll need to convert the datetime column 
#     into POSIXct before plotting.

# conversion of datetime col to POSIXct
# First I have to convert the datetime column from factor to character
bike[,"datetime"] = as.character(bike[,"datetime"])
# then I should be able to convert the time into that format
bike["datetime"]=as.POSIXct(bike[,"datetime"], format="%Y-%m-%d %H:%M:%OS") #the datetime column is a default date which can be called directly with this

pl <- ggplot(data=bike, aes(x=datetime ,y=count))
pl + geom_point(aes(color=factor(temp)))


# 6.  What is the correlation between temp and count? 
  
  # people use this service much more when it is warm outside.

# 7.  Let's explore the season data. Create a boxplot, 
#     with the y axis indicating count and the x axis begin a box for each season.

  # each boxplot corresponds to seasonal number(1:spring 2:summer 3:fall 4:winter)
  # It does not split it up by year since that is not part of the instructions
  # This data goes through 2 years with 4 seasons each, so it won't skew results
pl1 <- ggplot(data=bike, aes(factor(season), count))
pl1 + geom_boxplot()

# 8.  Create an "hour" column that takes the hour from the datetime column. 
#     You'll probably need to apply some function to the entire datetime 
#     column and reassign it. 
#     Hint:
#     time.stamp <- bike$datetime[4]
#     format(time.stamp, "%H")

bike$hour <- format(bike[,"datetime"], "%H") #This appends a column named hour that is derived from datetime


# 9.  Building the Model
#     Use lm() to build a model that predicts count based solely on the temp 
#     feature, name it temp.model

#split test/training data
library(caTools)
set.seed(101)
sample <- sample.split(bike$count, SplitRatio = 0.7)
train = subset(bike, sample == TRUE)
test = subset(bike, sample == FALSE)

temp.model <- lm(count ~ temp, data = train)

# 10. Get the summary of the temp.model
summary(temp.model)


# 11. How many bike rentals would we predict if the temperature was 25 
#     degrees Celsius? Calculate this two ways:
#     . Using the values we just got above
#     . Using the predict() function

#coefficent = 9.1320
#intercept = 8.792
#guess = intercept + coeff*temp
#my result is 237.1 (rounded to the nearest whole number, count = 237)

# 12. Use sapply() and as.numeric to change the hour column to a column of 
#     numeric values.
"hour column class before"
class(bike[["hour"]])
bike[["hour"]] <- sapply(bike[["hour"]], as.numeric)
"hour column class after"
class(bike[["hour"]])

# 13. Finally build a model that attempts to predict count based off of the 
#     following features. Figure out if theres a way to not have to 
#     pass/write all these variables into the lm() function. 
#     Hint: StackOverflow or Google may be quicker than the documentation.


#season
#holiday
#workingday
#weather
#temp
#humidity
#windspeed
#hour (factor)

bike[["hour"]] <- factor(bike[["hour"]])

#The data is already split, so I can directly call lm from here
model <- lm(count ~ season + holiday + workingday + temp + humidity + windspeed + hour, data = train)

# 14. Get the summary of the model
summary(model)

# 15. Did the model perform well on the training data? What do you think 
#     about using a Linear Model on this data? 
bike.predictions <- predict(model, test)
result <- cbind(bike.predictions, test$count)
colnames(result) <- c('predicted', 'actual')
result <- as.data.frame(result)
print(head(result))

  # This is an awful model the predicted outcomes are not even close to the actual values
  # a linear model doesn't reflect the correlation between some of the data so it throws the predictions off

#     You should have noticed that this sort of model doesn't work well 
#     given our seasonal and time series data. We need a model that can 
#     account for this type of trend, read about Regression Forests for 
#     more info if you're interested! For now, let's keep this in mind as a 
#     learning experience and move on towards classification with 
#     Logistic Regression!






