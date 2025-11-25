# Implementing Vowel Nasalization through Acoustics
# Julian Vargo
# Department of Spanish & Portuguese, UC Berkeley

# This script offers an intro to acoustical analysis of nasal vowels
# I (loosely) follow the methods in Carignan et al's (2023) paper on Arabana
  # https://doi.org/10.16995/labphon.9152
# Additional analysis and data is based on my (2025) HLS presentation

# Optional: clean up your environment and free up memory
rm(list=ls())
gc()

# Install packages if they're not already on your computer, then load them in.
for (p in c("dplyr","readr","stringr","xgboost","ggplot2","arrow","mgcv")) if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
invisible(lapply(c("dplyr","readr","stringr","xgboost","ggplot2","arrow","mgcv"), library, character.only = TRUE))

# Pull the df from GitHub - a download friendly alternative to read_csv()
tmp <- tempfile(fileext = ".parquet")
download.file("https://github.com/julian-vargo/nasalization-demo/raw/refs/heads/main/condensed.parquet", destfile = tmp, mode = "wb")
df <- arrow::read_parquet(tmp)
unlink(tmp)
df <- df %>% collect()

# # Optional: manual download
# df <- arrow::read_parquet("path/to/condensed.parquet")
# df <- df %>% collect()

View(df)

# If you only want to look at sonorants
# obstruents <- c("P","T","K","p","t","k")
# df <- df %>% filter(!preceding_phone %in% obstruents,
#                     !following_phone %in% obstruents)

# Let's make sure that our data is reproducible
set.seed(1)

# We must make sure that timestamp is a numeric factor or we might get errors
df <- df %>% mutate(timestamp = as.numeric(timestamp))

# Split out data into 20% prediction, 60% training, 20% validation
prediction_df <- df %>% slice_sample(prop = 0.2)
train_validate_df <- df %>% anti_join(prediction_df, by = names(df))
train_df <- train_validate_df %>% slice_sample(prop = 0.75)
validate_df <- train_validate_df %>% anti_join(train_df, by = names(train_validate_df))

# It's good practice to routinely keep our environment clean
rm(train_validate_df,p,tmp)

# We must predefine our nasal consonants
# lowercase is Spanish while uppercase is English
nasals <- c("m","n","ɲ", "ŋ", "M", "N", "NG")

# Let's predefine the acoutic features that we want in our model:
features <- c("f0","bandwidth1","bandwidth2","bandwidth3","harmonicity","intensity","cog","cogSD","skewness","kurtosis","a1p0")

# DISCUSSION:
# Why might we want to exclude formants from our model?

# We're going to train only on 10%-NV and 90%-VN sequences
# The 10% point of a vowel in a NV sequence will be nasalized
# The 90% point of a vowel in a VN sequence will be nasalized
# We use the edges of the vowels as our ground truth
train_df <- train_df %>% filter(
  train_df$timestamp == 10 |
  train_df$timestamp == 90) %>% filter(
  (following_phone %in% nasals & preceding_phone %in% nasals) |
  (!preceding_phone %in% nasals & !following_phone %in% nasals))

validate_df <- validate_df %>% filter(
  validate_df$timestamp == 10 |
    validate_df$timestamp == 90) %>% filter(
      (following_phone %in% nasals & preceding_phone %in% nasals) |
        (!preceding_phone %in% nasals & !following_phone %in% nasals))

train_df <- train_df %>%
  mutate(
    outcome = case_when(
      (following_phone %in% nasals & preceding_phone %in% nasals & timestamp == 90) |
      (preceding_phone %in% nasals & following_phone %in% nasals & timestamp == 10) ~ 1L, TRUE ~ 0L))

validate_df <- validate_df %>%
  mutate(
    outcome = case_when(
      (following_phone %in% nasals & timestamp == 90) |
        (preceding_phone %in% nasals & timestamp == 10) ~ 1L, TRUE ~ 0L))

# Train xgboost model
xgbtrain <- train_df %>%
  select(all_of(features)) %>%
  as.matrix() %>%
  xgb.DMatrix(label = train_df$outcome)

# When you set your parameters here, you should use cross validation!!!
# Cross validation is super important in machine learning,
# but it takes several hours/days to run much of the time.
# I cross validated to get the best parameters, and then
# turned down the nrounds so we still get a decent model for the lab
# but without the headache of long runtimes
xgb_model <- xgboost(
  data              = xgbtrain,
  objective         = "binary:logistic",
  eval_metric       = "logloss",
  eta               = 0.05,
  min_child_weight  = 1,
  max_depth         = 7,
  subsample         = 0.7,
  colsample_bytree  = 0.9,
  nrounds           = 100,
  verbose           = 0
)

xgbpredict <- prediction_df %>%
  select(all_of(features)) %>%
  as.matrix() %>%
  xgb.DMatrix()

# Now that we've trained our model, let's add P(Nasal) as a new column
# We call this outcome probability
# This is similar 
prediction_df <- prediction_df %>%
  mutate(outcome_probability = predict(xgb_model, xgbpredict))

# Validate
# same feature list you used for training
val_mat <- validate_df %>%
  dplyr::select(dplyr::all_of(features)) %>%
  as.matrix()

# P(outcome is 1, aka outcome is predicted to be nasal)
# This is conceptually similar to Carignan's "Predicted degree of nasalization". 
val_prob <- predict(xgb_model, val_mat)

# If our prediction probability is greater than 0.5 that the adjacent
# phone is nasal, we consider this a success
validate_df <- validate_df %>%
  dplyr::mutate(
    pred_prob  = val_prob,
    pred_label = as.integer(pred_prob > 0.5)  # threshold at 0.5
  )

# Let's examine our model performance
table(truth = validate_df$outcome, pred = validate_df$pred_label)
mean(validate_df$pred_label == validate_df$outcome)

# Plot variable importance
importance <- xgb.importance(model = xgb_model)
imp_df <- as.data.frame(importance)
ggplot(imp_df[1:11, ], aes(x = reorder(Feature, Gain), y = Gain, fill=Gain, color=Gain)) +
  geom_col() +
  coord_flip() +
  labs(x = "Acoustic Feature", y = "Importance (Model Gain)",
       title = "XGBoost Acoustic Predictors of Nasality") + theme_minimal()

# Now let's visualize nasalization tracks
ggplot(data=prediction_df, 
aes(x = timestamp,
y = outcome_probability,
colour = following_phone)) +
geom_smooth(
method  = "gam",
formula = y ~ s(x, k = 5),
se = TRUE
  ) +
  theme_minimal()

# AH! That is awful, let's condense our environments down
prediction_df <- prediction_df %>%
  mutate(
    environment = case_when(
      preceding_phone  %in% nasals & following_phone %in% nasals ~ "N_N",
      !(preceding_phone %in% nasals) & following_phone %in% nasals ~ "C_N",
      preceding_phone  %in% nasals & !(following_phone %in% nasals) ~ "N_C",
      !(preceding_phone %in% nasals) & !(following_phone %in% nasals) ~ "C_C",
      TRUE ~ NA))

# Do environments such as N_N, C_C, N_C, and C_N matter?
ggplot(data=prediction_df, 
       aes(x = timestamp,
           y = outcome_probability,
           colour = environment)) +
  geom_smooth(
    method  = "gam",
    formula = y ~ s(x, k = 5),
    se = TRUE
  ) +
  theme_minimal()

# Let's make a mega-factor, combining language and environment
prediction_df <- prediction_df %>%
  mutate(
    env_lang = case_when(
      environment == "N_N" & language == "e" ~ "english N_N",
      environment == "C_N" & language == "e" ~ "english C_N",
      environment == "N_C" & language == "e" ~ "english N_C",
      environment == "C_C" & language == "e" ~ "english C_C",
      environment == "N_N" & language == "s" ~ "spanish N_N",
      environment == "C_N" & language == "s" ~ "spanish C_N",
      environment == "N_C" & language == "s" ~ "spanish N_C",
      environment == "C_C" & language == "s" ~ "spanish C_C",
      TRUE ~ NA_character_))

# Does nasalization differ by environment AND by language?
ggplot(data=prediction_df, 
       aes(x = timestamp,
           y = outcome_probability,
           color = env_lang)) +
  geom_smooth(
    method  = "gam",
    formula = y ~ s(x, k = 5),
    se = TRUE) +
  theme_minimal() +
  labs(x = "Normalized time of vowel", y = "Degree of Nasalization", color = "Language & Environment", title = "Degree of Nasalization in Spanish and English Bilingual Vowels")
