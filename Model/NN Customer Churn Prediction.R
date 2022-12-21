library(tidyverse)
library(tidymodels)
library(tensorflow)
library(tfruns)
library(keras)
library(themis)

setwd(dirname(rstudioapi::getActiveDocumentContext()[[2]]))

# Read in full data
data_full <- read_csv("Telco_Churn_Data.csv") %>%
  janitor::clean_names() %>%
  mutate(
    senior_citizen = ifelse(senior_citizen == 1, "Yes", "No"),
    across(c(device_protection, online_backup, online_security,
             streaming_movies, streaming_tv, tech_support),
           ~ ifelse(.x == "No internet service", "No", .x)),
    across(c(multiple_lines),
           ~ ifelse(.x == "No phone service", "No", .x)),
    across(where(is.character), as.factor),
    churn = fct_rev(churn)
  ) %>%
  drop_na()

# Initial splits
set.seed(1)
dt_split <- data_full  %>%
  mutate(churn = ifelse(churn == "Yes", 1, 0) %>% as.factor() %>% fct_rev()) %>%
  initial_split(strata = "churn")

dt_train <- training(dt_split)
dt_test <- testing(dt_split)

# Apply preprocessing separately
preproc_recipe <- recipe(churn ~ ., data = dt_train) %>%
  step_rm(customer_id) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(churn, skip = TRUE)

# Preprocess data
dt_train_prepped <- preproc_recipe %>% prep() %>% juice()
dt_test_prepped <- preproc_recipe %>% prep() %>% bake(new_data = dt_test)


dt_train_x <- dt_train_prepped %>% select(-churn) %>% as.matrix
dt_train_y <- dt_train_prepped %>% select(churn) %>% as.matrix %>% to_categorical()

dt_test_x <- dt_test_prepped %>% select(-churn) %>% as.matrix
dt_test_y <- dt_test_prepped %>% select(churn) %>% as.matrix %>% to_categorical()

# Hyperparameters
hyperparams <- crossing(
  units2 = c(2^4, 2^5, 2^6, 2^7),
  units3 = c(2^4, 2^5, 2^6, 2^7),
  batch_size = c(20, 50, 100),
  learning_rate = c(1e-5, 1e-4, 1e-3),
)

# Tuning run
runs <- tuning_run(file = "Keras Train.R", runs_dir = "ANN_runs",
                   flags = hyperparams, echo = FALSE, confirm = FALSE)

# View results
runs_df <- ls_runs(runs_dir = "ANN_runs/")

# View best run
view_run(run_dir = runs_df %>% 
           slice_max(metric_val_auc, n = 1), 
         viewer = getOption("tfruns.viewer"))

# Compare runs
compare_runs(runs = runs_df %>% 
               slice_max(metric_val_auc, n = 2),
             viewer = getOption("tfruns.viewer"))

# Select the best model
best_run <- runs_df %>% 
  # filter(epochs_completed > 10) %>% 
  arrange(metric_val_loss) %>% 
  head(1)

best_run

# Train the model with best parameter
run <- training_run("Keras Train.R", 
                    flags = list(
                      units2 = best_run$flag_units2,
                      units3 = best_run$flag_units3,
                      batch_size = best_run$flag_batch_size,
                      learning_rate = best_run$flag_learning_rate
                    ))

# Load best model
best_model <- load_model_hdf5("ann_model.h5")

# Evaluate the model
model %>% evaluate(dt_test_x, dt_test_y)

# Evaluation metrics
eval_metrics <- metric_set(accuracy, sensitivity, specificity, precision,
                           recall)

model %>% 
  predict(dt_test_x) %>% 
  as_tibble() %>% 
  bind_cols(dt_test_y) %>% 
  transmute(.pred_Yes = V2, .pred_No = V1,
            .pred = ifelse(V2 >= 0.5, "Yes", "No") %>% as.factor,
            churn = ifelse(...4 == 1, "Yes", "No") %>% as.factor) %>% 
  eval_metrics(truth = churn, estimate = .pred)

# Profit Curves and maximum profit
case_counts_fit <- function(final_fit, probs, holdout_data, truth){
  bind_cols(
    final_fit %>% 
      predict(holdout_data) %>% 
      as_tibble() %>% 
      rename(.pred_Yes = V2, .pred_No = V1),
    truth %>% 
      transmute(churn = ifelse(churn == 1, "Yes", "No") %>% as.factor)
  ) %>% 
    mutate(
      .pred_thr = ifelse(.pred_Yes > probs, "Yes", "No"),
      case = case_when(
        .pred_thr == "Yes" & churn == "Yes" ~ "TP",
        .pred_thr == "Yes" & churn == "No" ~ "FP",
        .pred_thr == "No" & churn == "Yes" ~ "FN",
        .pred_thr == "No" & churn == "No" ~ "TN"
      )) %>% 
    count(case)  
}

profit_curve_fit <- function(final_fit, probs, holdout_data, truth){
  probs %>% 
    as_tibble() %>%
    rename(threshold = value) %>%
    mutate(counts = map(threshold, ~ case_counts_fit(final_fit, .x,
                                                     holdout_data, truth))) %>%
    unnest(counts) %>%
    pivot_wider(values_from = n, names_from = case) %>% 
    mutate(
      across(everything(), ~ replace(., is.na(.), 0)),
      profit_without_model = TP*0 + FP*500 + TN*500 + FN*0,
      profit_with_model = TP*500*0.666*0.5 + FP*500*0.666 + TN*500 + FN*0,
      value_add = profit_with_model - profit_without_model,
      sign = ifelse(value_add > 0, "positive", "negative")
    )
}

ann_profits <- profit_curve_fit(final_fit = best_model,
                                probs = seq(0, 1, 0.05),
                                holdout_data = dt_test_x,
                                truth = dt_test)

ann_profits %>% 
  ggplot(aes(threshold, profit_with_model)) +
  geom_line(colour = "grey50", size = 0.4) + 
  geom_point(aes(colour = sign, group = 1)) +
  labs(colour = "Value-add of the \nmodel compared \nto using no model:",
       y = "Profit",
       x = "Classification Threshold", 
       title = "Forecasted Profit Depending On Classification Threshold",
       subtitle = "Artifical Neural Network") +
  scale_y_continuous(labels = dollar_format()) +
  scale_x_continuous(labels = percent_format(), 
                     breaks = seq(0, 1, 0.1)) +
  scale_colour_manual(values = c("firebrick", "dodgerblue")) +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(face = "italic", colour = "grey50",
                                     size = 12))

# Extracting maximum profit
ann_profits %>% 
  slice_max(profit_with_model, n = 1) %>% 
  transmute(threshold,
            profit_delta = profit_with_model/profit_without_model - 1,
            value_add)
