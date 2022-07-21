## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width=6, fig.height=4
)

## ----setup--------------------------------------------------------------------
library(ensModelVis)

## ---- include = FALSE---------------------------------------------------------
if (rlang::is_installed("stacks") && 
    rlang::is_installed("tidymodels") && 
    rlang::is_installed("kernlab")) {
  run <- TRUE
} else {
  run <- FALSE
}

knitr::opts_chunk$set(
  eval = run
)

## ----libs, message=FALSE------------------------------------------------------
library(tidymodels)
library(stacks)

## -----------------------------------------------------------------------------
data("mtcars")
mtcars <- mtcars %>% mutate(cyl = as.factor(cyl), vs = as.factor(vs), am = as.factor(am))

## -----------------------------------------------------------------------------

set.seed(1)
mtcars_split <- initial_split(mtcars)
mtcars_train <- training(mtcars_split)
mtcars_test  <- testing(mtcars_split)

set.seed(1)
folds <- vfold_cv(mtcars_train, v = 5)

mtcars_rec <- 
  recipe(mpg ~ ., 
         data = mtcars_train)

metric <- metric_set(rmse)

ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

## -----------------------------------------------------------------------------
# LINEAR REG
lin_reg_spec <-
  linear_reg() %>%
  set_engine("lm")

# extend the recipe
lin_reg_rec <-
  mtcars_rec %>%
  step_dummy(all_nominal()) 

# add both to a workflow
lin_reg_wflow <- 
  workflow() %>%
  add_model(lin_reg_spec) %>%
  add_recipe(lin_reg_rec)

# fit to the 5-fold cv
set.seed(2020)
lin_reg_res <- 
  fit_resamples(
    lin_reg_wflow,
    resamples = folds,
    metrics = metric,
    control = ctrl_res
  )

# SVM
svm_spec <- 
  svm_rbf(
    cost = tune("cost"), 
    rbf_sigma = tune("sigma")
  ) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

# extend the recipe
svm_rec <-
  mtcars_rec %>%
  step_dummy(all_nominal()) %>%
  step_impute_mean(all_numeric(), skip = TRUE) %>%
  step_corr(all_predictors(), skip = TRUE) %>%
  step_normalize(all_numeric(), skip = TRUE)

# add both to a workflow
svm_wflow <- 
  workflow() %>% 
  add_model(svm_spec) %>%
  add_recipe(svm_rec)

# tune cost and sigma and fit to the 5-fold cv
set.seed(2020)
svm_res <- 
  tune_grid(
    svm_wflow, 
    resamples = folds, 
    grid = 6,
    metrics = metric,
    control = ctrl_grid
  )


## -----------------------------------------------------------------------------
mtcars_model_st <- 
  stacks() %>%
  add_candidates(lin_reg_res) %>%
  add_candidates(svm_res) %>%
  blend_predictions() %>%
  fit_members()

## -----------------------------------------------------------------------------
member_preds <- 
  mtcars_test %>%
  select(mpg) %>%
  bind_cols(predict(mtcars_model_st, mtcars_test, members = TRUE))

## -----------------------------------------------------------------------------
map_dfr(member_preds, rmse, truth = mpg, data = member_preds) %>%
  mutate(member = colnames(member_preds))

## -----------------------------------------------------------------------------
p1 <- plot_ensemble(truth = member_preds$mpg, tibble_pred = member_preds %>% select(-mpg))
p1 + geom_abline()

plot_ensemble(truth = member_preds$mpg, tibble_pred = member_preds %>% select(-mpg), facet = TRUE)

