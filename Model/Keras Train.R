# Specifying flags for hyperparameter tuning: Initialise random values
FLAGS <- flags(
  flag_numeric("units2", 2^6),
  flag_numeric("units3", 2^6),
  flag_numeric("batch_size", 50),
  flag_numeric("learning_rate", 0.001)
)

# Specify model
model <- keras_model_sequential() %>% 
  layer_dense(units = ncol(dt_train_x), activation = "relu", 
              input_shape = ncol(dt_train_x)) %>% 
  layer_dense(units = FLAGS$units2, activation = 'relu') %>%
  layer_dense(units = FLAGS$units3, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

model %>% 
  compile(optimizer = optimizer_adam(learning_rate = FLAGS$learning_rate),
          loss = "binary_crossentropy",
          metrics = tf$keras$metrics$AUC())

# Define early stop
early_stop <- callback_early_stopping(monitor = "val_auc", patience = 10,
                                      mode = "max")

# Fit the model
fit_hist <- model %>% 
  fit(
    dt_train_x,
    dt_train_y,
    epochs = 100,
    batch_size = FLAGS$batch_size,
    validation_split = 0.3,
    verbose = 1,
    view_metrics = FALSE,
    callbacks = list(early_stop)
  )

# Evaluate model on holdout
model %>% 
  evaluate(dt_test_x, dt_test_y)

# Save the model
save_model_hdf5(model, "ann_model.h5")