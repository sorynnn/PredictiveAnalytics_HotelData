# Libraries
library(tidyverse)
library(caret)
library(car)
library(skimr)
install.packages("DataExplorer")
library(DataExplorer)
install.packages("ROSE")
library(ROSE)


# Step 0: EDA
OC_data_original <- read_csv("~/Downloads/OceanCrestdata.csv")

# OC_data <- read_csv("OceanCrest_subset.csv",
#     col_types = cols(
#       is_canceled = col_factor(levels = c("0", "1")),
#                     #hotel = col_factor(levels = c("Resort Hotel", "City Hotel")),
#                     lead_time = col_double(),
#                     arrival_date_year = col_integer(),
#                     arrival_date_week_number = col_integer(),
#                     arrival_date_day_of_month = col_integer(),
#                     adults = col_integer(), babies = col_integer(),
#                     previous_cancellations = col_integer(),
#                     previous_bookings_not_canceled = col_integer(),
#                     booking_changes = col_integer(),
#                     days_in_waiting_list = col_integer(),
#                     required_car_parking_spaces = col_integer(),
#                     total_of_special_requests = col_integer()
#       )
#     )

summaryStats <- skim(OC_data)
summaryStats

create_report(OC_data, y = "is_canceled")

table(OC_data$is_canceled)
table(OC_data_original$stays_in_weekend_nights,
      OC_data_original$stays_in_week_nights)


#Step 1 - Partition our Data and Pre-processing
#0
OC_data <- OC_data_original %>%
              select(-required_car_parking_spaces, -arrival_date_day_of_month) %>%
              mutate(babies = case_when(
                                babies >= 1 ~ 1,
                                TRUE ~ 0
                     ),
                     booking_changes = case_when(
                       booking_changes >= 1 ~ 1,
                       TRUE ~ 0
                     ),
                     special_requests = case_when(
                       total_of_special_requests >= 1 ~ 1,
                       TRUE ~ 0
                     ),
                     cancellation_rate = previous_cancellations/(
                       previous_cancellations + previous_bookings_not_canceled
                       ),
                     weekend_percent = stays_in_weekend_nights/(
                       stays_in_weekend_nights + stays_in_week_nights
                     ),
                     stay_days = (
                       stays_in_weekend_nights + stays_in_week_nights
                     ),
                     across(
                       c(
                         lead_time, adults, adr, days_in_waiting_list
                       ),
                       ~scale(.)[,1]
                     ),
                     # cancellation_rate = if_else(is.nan(cancellation_rate), 
                     #                             0, cancellation_rate)
                     across(
                       c(cancellation_rate, weekend_percent), ~ if_else(is.nan(.), 
                                      0, .)
                     )
                     
                    ) %>%
              select(-previous_cancellations, -previous_bookings_not_canceled, 
                     -stays_in_weekend_nights, -stays_in_week_nights,
                     -total_of_special_requests
                     ) %>%
              select(-is_canceled, everything(), is_canceled)

#1. change reponse and categorical variables to factor
OC_data <- OC_data %>% 
  mutate_at(c("is_canceled", "babies", "booking_changes", "special_requests",
              "is_repeated_guest"
              ), 
            as.factor)

#2. rename resonse 
OC_data$is_canceled<-fct_recode(OC_data$is_canceled, is_canceled = "1",not_is_canceled = "0")

#3. relevel response
OC_data$is_canceled<- relevel(OC_data$is_canceled, ref = "is_canceled")

#make sure levels are correct
levels(OC_data$is_canceled)

OC_predictors_dummy <- model.matrix(is_canceled~ ., data = OC_data)#create dummy variables expect for the response
OC_predictors_dummy<- data.frame(OC_predictors_dummy[,-1]) #get rid of intercept
OC_data <- cbind(OC_predictors_dummy, is_canceled = OC_data$is_canceled)

#index
# Train/Test split before SMOTE
set.seed(99) #set random seed
index <- createDataPartition(OC_data$is_canceled, p = .8,list = FALSE)
OC_train <- OC_data[index,]
OC_test <- OC_data[-index,]

# SMOTE
# Apply SMOTE to balance the classes
OC_train_smote <- ovun.sample(is_canceled ~ ., data = OC_train, method = "over", p = 0.5, seed = 1234)$data

# Check the class distribution after applying SMOTE
table(OC_train_smote$is_canceled)


# Export data to CSV
write.csv(OC_data, file = "OC_data.csv", row.names = FALSE)
write.csv(OC_train, file = "OC_train.csv", row.names = FALSE)
write.csv(OC_test, file = "OC_test.csv", row.names = FALSE)
write.csv(OC_train_smote, file = "OC_train_smote.csv", row.names = FALSE)


#Parallel Processing
#install.packages("xgboost")
library(xgboost)

install.packages("doParallel")
library(doParallel)

#total number of cores on your computer
num_cores<-detectCores(logical=FALSE)
num_cores

#start parallel processing w/ XGBoost
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

colSums(is.na(OC_train))
OC_train <- na.omit(OC_train)

set.seed(8)
model_gbm <- train(is_canceled~.,
                   data = OC_train,
                   method = "xgbTree",
                   # provide a grid of parameters
                   tuneGrid = expand.grid(
                     nrounds = c(50,200),
                     eta = c(0.025, 0.05),
                     max_depth = c(2, 3),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   trControl= trainControl(method = "cv",
                                           number = 5,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary),
                   metric = "ROC"
)
#stop parallel processing
stopCluster(cl)
registerDoSEQ()

#Performance based on various tuning parameters
plot(model_gbm)

#Print out of the best tuning parameters
model_gbm$bestTune

#only print top 10 important variables
plot(varImp(model_gbm), top=10)

#SHAP
#install.packages("SHAPforxgboost")
library(SHAPforxgboost)

# Exclude character columns and is_canceled


Xdata<-as.matrix(select(OC_train,-is_canceled)) # change data to matrix for plots

Xdata <- OC_train %>% 
  select_if(is.numeric) %>% 
  as.matrix()

# Calculate SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot for top 15 variables
shap.plot.summary.wrap1(model_gbm$finalModel, X = Xdata, top_n = 10)

#example partial dependence plot
##Change this shit
p <- shap.plot.dependence(
  shap, 
  x = "TRAN_AMT", #top val in shapp
  color_feature = "CUST_AGE", 
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) +
  ggtitle("TRAN_AMT")
print(p)




