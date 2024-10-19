library(class)
library(ggplot2)
library(caret)
library(lattice)
library(tidyverse)
library(RColorBrewer)
library(tm)
library(slam)
library(e1071)
library(ISLR)
library(magrittr)
library(dplyr)
library(fastDummies)
library(mice)
library(car)
library(stats)
rawDF <- read_csv("https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/data-mining-s2y2223-AnitaMunar/master/1652126webshop.csv")
head(rawDF)
str(rawDF)
webshop_data <- read_csv("https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/data-mining-s2y2223-AnitaMunar/master/1652126webshop.csv")

#part1a - missing values
webshop_data <- na.omit(webshop_data)


#Cook's D remove outliers

# Define predictor variables
predictor_vars <- c("Age", "Shipping_Time", "Pictures", "Find_website", "Device", "Time_Spent_on_Website", "Number_of_products_browsed", "Ease_of_purchase", "Review_rating")

# Initialize empty list to store outliers for each variable
outliers_list <- vector("list", length(predictor_vars))
names(outliers_list) <- predictor_vars

# Loop through each predictor variable and compute Cook's distance
for (var in predictor_vars) {
  
  # Create linear regression model
  lm_model <- lm(webshop_data$Purchase_Amount ~ . - Purchase_Amount, data = webshop_data)
  
  # Compute Cook's distance for each observation
  cooks_d <- cooks.distance(lm_model)
  
  # Identify outliers using Cook's distance threshold of 4/n
  n <- nrow(webshop_data)
  outliers <- cooks_d > 4/n
  
  # Store outliers for current variable in list
  outliers_list[[var]] <- webshop_data[outliers, ]
  
  # Print results
  cat(paste0("Outliers for ", var, ":\n"))
  print(outliers_list[[var]])
  cat("\n")
}
#remove outliers
outliers_df <- do.call(rbind, outliers_list)
webshop_data <- anti_join(webshop_data, outliers_df)

#create dummies
webshop_data <- webshop_data %>%
  mutate(Device_dummy = case_when(Device == "Mobile" ~ 1,
                                  Device == "PC" ~ 2),
         Find_website_dummy = case_when(Find_website == "Social_Media_Advertisement" ~ 1,
                                        Find_website == "Search_Engine" ~ 2,
                                        Find_website == "Friends_or_Family" ~ 3,
                                        Find_website == "Other" ~ 4)) %>%
  select(-Device, -Find_website)

#part1c
cor_matrix <- cor(webshop_data[, c("Time_Spent_on_Website", "Number_of_products_browsed","Review_rating", "Ease_of_purchase", "Find_website_dummy", "Pictures", "Device_dummy")])
print(cor_matrix)

webshop_data$Time_Products_Average <- (webshop_data$Time_Spent_on_Website/60 + webshop_data$Number_of_products_browsed)/2

webshop_data$Time_Spent_on_Website <- NULL
webshop_data$Number_of_products_browsed <- NULL



#part1d
# Scatter plot of Age vs. Purchase_Amount
plot(webshop_data$Age, webshop_data$Purchase_Amount)

# Scatter plot of Shipping_Time vs. Purchase_Amount
plot(webshop_data$Shipping_Time, webshop_data$Purchase_Amount)

# Scatter plot of Pictures vs. Purchase_Amount
plot(webshop_data$Pictures, webshop_data$Purchase_Amount)

ggplot(webshop_data, aes(x = Find_website_dummy, y = Purchase_Amount)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  labs(x = "Find Website Dummy", y = "Purchase Amount")

ggplot(webshop_data, aes(x = Device_dummy, y = Purchase_Amount)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE)

webshop_data <- webshop_data %>%
  mutate(Age_squared = Age^2) %>%
  select(-Age) %>%
  rename(Age = Age_squared)

# Plot the lowess curve
ggplot(webshop_data, aes(x = Age, y = Purchase_Amount)) +
  geom_point() +                  # scatter plot of the data
  geom_smooth(method = "lowess")  # lowess curve


#2a
summary(lm(Purchase_Amount~Time_Products_Average,data=webshop_data))
summary(lm(Purchase_Amount~Pictures,data=webshop_data))
summary(lm(Purchase_Amount~Shipping_Time,data=webshop_data))
summary(lm(Purchase_Amount~Review_rating,data=webshop_data))
summary(lm(Purchase_Amount~Ease_of_purchase,data=webshop_data))
summary(lm(Purchase_Amount~Age,data=webshop_data))
summary(lm(Purchase_Amount~Device_dummy,data=webshop_data))
summary(lm(Purchase_Amount~Find_website_dummy,data=webshop_data))



model_a <- lm(Purchase_Amount ~ Time_Products_Average+Review_rating+Ease_of_purchase+Pictures+Find_website_dummy+Shipping_Time+Device_dummy, data = webshop_data)
summary(model_a)

#2b
scaled_data <- webshop_data
scaled_data[, c("Time_Products_Average", "Review_rating", "Ease_of_purchase")] <- scale(scaled_data[, c("Time_Products_Average", "Review_rating", "Ease_of_purchase")])
model_b <- lm(Purchase_Amount ~ Time_Products_Average+Review_rating+Ease_of_purchase+Find_website_dummy+Device_dummy, data = scaled_data)
summary(model_b)

# Compare the models
summary(model_a)
summary(model_b)

#age model
model_age <- lm(Purchase_Amount ~ Age + I(Age^2), data = webshop_data)
summary(model_age)

#2c
# Fit the linear regression model
model_c <- lm(Purchase_Amount ~ Time_Products_Average + Review_rating + Ease_of_purchase, data = webshop_data)
summary(model_c)


new_customer <- data.frame(Time_Products_Average = 12.05, Review_rating = 4.5, Ease_of_purchase = 4)

predicted_purchase_amount <- predict(model_c, newdata = new_customer)
print(predicted_purchase_amount)


# calculate the mean squared error (MSE) of the model using cross-validation
mse <- train(Purchase_Amount ~ Time_Products_Average + Review_rating + Ease_of_purchase, data = webshop_data, method = "lm", trControl = trainControl("cv", 10), metric = "RMSE")

# print the mean squared error
print(mse)


# extract the residuals from the model
residuals <- resid(model_c)
# create a normal probability plot of the residuals
qqnorm(residuals)
qqline(residuals)
print(residuals)

#predict based on age model
new_customer_age <- data.frame(Age = 35)
new_customer_age$Age_sq <- new_customer_age$Age^2
predicted_purchase_amount_age <- predict(model_age, newdata = new_customer_age)
print(predicted_purchase_amount_age)


#part2d
#part2da
#remove negative values
webshop_missing_data <- rawDF

webshop_missing_data <- webshop_missing_data %>%
  filter(Time_Spent_on_Website >= 0 & Number_of_products_browsed >= 0)

# Remove outliers, dummies and combining multicollinear columns
boxplot(webshop_missing_data[,c(2,3,4,5,6,8,9)])

webshop_missing_data <- webshop_missing_data %>%
  mutate(Find_website_dummy_2 = if_else(is.na(Find_website), NA_integer_, 
                                        recode(Find_website, 
                                               `Social_Media_Advertisement` = 1, 
                                               `Search_Engine` = 2, 
                                               `Friends_or_Family` = 3, 
                                               `Other` = 4))) %>%
  select(-Find_website)

# Create a copy of the original dataset
webshop_imputed <- webshop_missing_data

# Set seed for reproducibility
set.seed(123)

# Perform regression imputation
webshop_imputed <- mice(webshop_imputed, m=5, method="norm.predict", seed=123)

# Extract the imputed data
webshop_imputed <- complete(webshop_imputed)


#the imputation causes the dummy variable to leave the given dimensions of integers and become a float value. this makes it
#harder to interpret.



#part2dd
# Scatter plot of Age vs. Purchase_Amount
plot(webshop_imputed$Age, webshop_missing_data$Purchase_Amount)

# Scatter plot of Shipping_Time vs. Purchase_Amount
plot(webshop_imputed$Shipping_Time, webshop_missing_data$Purchase_Amount)

# Scatter plot of Pictures vs. Purchase_Amount
plot(webshop_imputed$Pictures, webshop_missing_data$Purchase_Amount)


#make average of 2 columns
webshop_imputed$Time_Products_Average <- (webshop_imputed$Time_Spent_on_Website/60 + webshop_imputed$Number_of_products_browsed)/2

webshop_imputed$Time_Spent_on_Website <- NULL
webshop_imputed$Number_of_products_browsed <- NULL




ggplot(webshop_imputed, aes(x = Find_website_dummy_2, y = Purchase_Amount)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  labs(x = "Find Website Dummy", y = "Purchase Amount")


#2a
summary(lm(Purchase_Amount~Time_Products_Average,data=webshop_imputed))
summary(lm(Purchase_Amount~Pictures,data=webshop_imputed))
summary(lm(Purchase_Amount~Shipping_Time,data=webshop_imputed))
summary(lm(Purchase_Amount~Review_rating,data=webshop_imputed))
summary(lm(Purchase_Amount~Ease_of_purchase,data=webshop_imputed))
summary(lm(Purchase_Amount~Age,data=webshop_imputed))
summary(lm(Purchase_Amount~Device,data=webshop_imputed))
summary(lm(Purchase_Amount~Find_website_dummy_2,data=webshop_imputed))

model_d <- lm(Purchase_Amount ~ Time_Products_Average+Review_rating+Ease_of_purchase+Pictures+Find_website_dummy_2+Shipping_Time+Device, data = webshop_imputed)
summary(model_d)

#2b
scaled_data2 <- webshop_imputed
scaled_data2[, c("Time_Products_Average", "Review_rating", "Ease_of_purchase")] <- scale(scaled_data2[, c("Time_Products_Average", "Review_rating", "Ease_of_purchase")])

model_e <- lm(Purchase_Amount ~ Time_Products_Average+Review_rating+Ease_of_purchase+Find_website_dummy+Device, data = scaled_data2)
summary(model_e)

# Compare the models
summary(model_c)
summary(model_d)
summary(model_e)

