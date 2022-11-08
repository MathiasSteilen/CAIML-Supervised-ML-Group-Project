library(tidyverse)
library(tidymodels)
library(keras)
library(modeltime)
library(timetk)

setwd(paste0(rprojroot::find_rstudio_root_file(),
             "/Model"))


data <- read_csv("Telco_Churn_Data.csv")

data_prepped <- recipe(churn ~ .,
                       data = dt_train) %>%
  step_rm(customer_id) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(churn, skip = TRUE) %>% 
  prep() %>% 
  juice()

# Splitting the data
dt_split <- data_prepped %>% 
  mutate(churn = ifelse(churn == "Yes", 1, 0)) %>% 
  initial_split(strata = "churn")

dt_train <- training(dt_split)
dt_test <- testing(dt_split)

dt_train_x <- dt_train %>% select(-churn) %>% as.matrix
dt_train_y <- dt_train %>% select(churn) %>% as.matrix %>% to_categorical()

dt_test_x <- dt_test %>% select(-churn) %>% as.matrix
dt_test_y <- dt_test %>% select(churn) %>% as.matrix %>% to_categorical()

# Fitting the ANN
model <- keras_model_sequential() %>% 
  layer_dense(units = ncol(dt_train_x), activation = "relu", 
              input_shape = ncol(dt_train_x)) %>% 
  layer_dense(units = 2^8, activation = 'relu') %>%
  layer_dense(units = 2^8, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'sigmoid')

model %>% 
  compile(optimizer = "adam",
          loss = "binary_crossentropy",
          metrics = "accuracy")

fit_hist <- model %>% 
  fit(
    dt_train_x,
    dt_train_y,
    epochs = 50,
    batch_size = 500,
    validation_split = 0.2
  )

# Plot training history
fit_hist$metrics %>% 
  as_tibble() %>% 
  select(loss, val_loss) %>% 
  mutate(epoch = 1:nrow(.)) %>% 
  pivot_longer(-epoch) %>% 
  ggplot(aes(epoch, value, colour = name)) +
  geom_line(size = 0.75) +
  geom_point() +
  # scale_y_log10(labels = scales::comma_format()) +
  labs(y = "accuracy",
       title = "Training History",
       colour = NULL) +
  theme_bw()

# Evaluate the model
model %>% evaluate(dt_test_x, dt_test_y)

model %>% 
  predict(dt_test_x) %>% 
  as_tibble() %>% 
  bind_cols(dt_test_y) %>% 
  rename(.pred = V1)
  accuracy(truth = churn, estimate = .pred)

model %>% 
  predict(dt_test_x) %>% 
  as_tibble() %>% 
  bind_cols(dt_test_y) %>% 
  rename(.pred_pos = V1, .pred_neg = V2, class_pos)
  count(`...3`)
  mape(truth = mwh, estimate = .pred)

# Making predictions
model %>% 
  predict(dt_test_x) %>% 
  as_tibble() %>% 
  rename(.pred = V1) %>% 
  bind_cols(dt_test_y) %>% 
  ggplot(aes(mwh, .pred)) +
  geom_point(alpha = 0.3, colour = "midnightblue") +
  geom_abline(colour = "grey25", size = 0.75, lty = "dashed") +
  theme_bw()

# Comparing to linear model baseline
workflow() %>% 
  add_model(
    linear_reg() %>% 
      set_engine("lm")
  ) %>% 
  add_recipe(
    recipe(mwh ~ ., data = dt_train)
  ) %>% 
  fit(dt_train) %>% 
  augment(dt_test) %>%
  mape(truth = mwh, estimate = .pred)

workflow() %>% 
  add_model(
    linear_reg() %>% 
      set_engine("lm")
  ) %>% 
  add_recipe(
    recipe(mwh ~ ., data = dt_train)
  ) %>% 
  fit(dt_train) %>% 
  augment(dt_test) %>%
  ggplot(aes(mwh, .pred)) +
  geom_point(alpha = 0.3, colour = "midnightblue") +
  geom_abline(colour = "grey25", size = 0.75, lty = "dashed") +
  theme_bw()

