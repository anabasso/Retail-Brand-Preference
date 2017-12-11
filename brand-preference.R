# LIBRARIES ----
library(reshape2)
library(ggplot2)
library(ggvis)
library(caret)
library(C50)


# DATA ----
dt <- read.csv("SComplete.csv")
head(dt)
str(dt)

dt$elevel <- factor(dt$elevel, ordered = TRUE)
dt$car <- factor(dt$car)
dt$zipcode <- factor(dt$zipcode)
dt$brand <- factor(dt$brand, c(0, 1), c("Acer","Sony"))

pred <- read.csv("SComplete.csv")
head(pred)
str(pred)

pred$elevel <- factor(pred$elevel, ordered = TRUE)
pred$car <- factor(pred$car)
pred$zipcode <- factor(pred$zipcode)
pred$brand <- factor(pred$brand, c(0, 1), c("Acer","Sony"))

# COMPARE DATA STRUCTURE FROM TEST AND PREDICT ----
summary(dt)
summary(pred)

# DATA DISTRIBUTION ----
# Plot brand and salary
plot(dt$brand, dt$salary)

# DATA VISUALIZATIONS ---- 
# Plot Matrix with 3 dimensions
pairs(dt[1:6], pch = 21, bg = c("red", "blue")[unclass(dt$brand)])

# Plot Age by Salary and Brand Preference
dt %>% 
  ggvis(~age, ~salary, fill = ~brand) %>% 
  layer_points()

# MODEL BUILDING ----
# Create train and test sets
set.seed(107)
inTrain <- createDataPartition(y = dt$brand,
                               p = .7,
                               list = FALSE)

train <- dt[inTrain,]
test  <- dt[-inTrain,]
nrow(train)
nrow(test)

# Create Train Control
trctrl <- trainControl(method = "repeatedcv", 
                       number = 10, 
                       repeats = 1)

# KNN ----
knn_fit <- train(brand ~., 
                 data = train, 
                 method = "knn",
                 trControl = trctrl,
                 preProcess = c("center", "scale"))

knn_fit
plot(knn_fit)

test$knn <- predict(knn_fit, newdata = test)

# Evaluating the model
postResample(test$knn, test$brand)
confusionMatrix(test$knn, test$brand)

test$knn_prob <- predict(knn_fit, newdata = test, type = "prob")

# Random Forest ----
rf_fit <- train(brand ~., 
                data = train, 
                method = "rf",
                trControl = trctrl)

rf_fit
plot(rf_fit)

test$rf <- predict(rf_fit, newdata = test)

# Evaluating the model
postResample(test$rf, test$brand)
confusionMatrix(test$rf, test$brand)

# C5: Caret ----
c5_fit <- train(brand ~., 
                 data = train, 
                 method = "C5.0",
                 trControl = trctrl,
                 preProcess = c("center", "scale"))

c5_fit
plot(c5_fit)

test$c5 <- predict(c5_fit, newdata = test)

# Evaluating the model
postResample(test$c5, test$brand)
confusionMatrix(test$c5, test$brand)


# C5 ----
c5b_fit <- C5.0(brand ~., 
                data = train,
                trials = 20,
                model = rules,
                winnow = FALSE)

c5b_fit
summary(c5b_fit)

test$c5b <- predict(c5b_fit, newdata = test)

# Evaluating the model
postResample(test$c5b, test$brand)
confusionMatrix(test$c5b, test$brand)

# resamps <- resamples(list(knn = knn_fit,  rf = rf_fit, c5_caret = c5_fit, c5 = c5b_fit))
# summary(resamps)

