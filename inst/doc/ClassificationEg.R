## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width=6, fig.height=4
)

## ---- include = FALSE---------------------------------------------------------
if (rlang::is_installed("stacks") && 
    rlang::is_installed("stringr") && 
    rlang::is_installed("discrim") && 
    rlang::is_installed("tidymodels") && 
    rlang::is_installed("glmnet") && 
    rlang::is_installed("nnet") &&
    rlang::is_installed("ranger") &&
    rlang::is_installed("MASS")) {
  run <- TRUE
} else {
  run <- FALSE
}
knitr::opts_chunk$set(
  eval = run
)

## ----setup--------------------------------------------------------------------
library(ensModelVis)

## ----load_libraries, message=FALSE--------------------------------------------
library(tidymodels)
library(stacks)
library(stringr)
library(discrim) # for LDA

## ----load_tr_data-------------------------------------------------------------
data(iris)

train <- iris %>% rename(Response = Species) %>% relocate(Response)


set.seed(1979)


tr <- initial_split(train, prop = .5, strata = Response)
train_data <- training(tr)
test_data  <- testing(tr)

mn <- apply(train_data %>% select(-Response), 2, mean)
sd <- apply(train_data %>% select(-Response), 2, sd)

train_data[,-c(1)] <- sweep(train_data[,-c(1)] ,2, mn, "-")
train_data[,-c(1)] <- sweep(train_data[,-c(1)] ,2, sd, "/")

test_data[,-c(1)] <- sweep(test_data[,-c(1)] ,2, mn, "-")
test_data[,-c(1)] <- sweep(test_data[,-c(1)] ,2, sd, "/")


## -----------------------------------------------------------------------------
spec_rec <- recipe(Response ~ ., data = train_data)
spec_wflow <- 
  workflow() %>%
  add_recipe(spec_rec)

ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()


folds <- train_data %>% vfold_cv(v = 10, strata = Response)

## ----eval = TRUE--------------------------------------------------------------
nnet_mod <-
  mlp(hidden_units = tune(), 
      penalty = tune(), 
      epochs = tune()
      ) %>%
  set_mode("classification") %>%
  set_engine("nnet")


nnet_wf <- 
  spec_wflow %>%
  add_model(nnet_mod)

nnet_res <- 
  nnet_wf %>%
  tune_grid(
    resamples = folds, 
    grid = 10,
    control = ctrl_grid
  )



# ===================================


lasso_reg_grid <- tibble(penalty = 10^seq(-8, -1, length.out = 10))



en_mod <- 
  multinom_reg(penalty = tune(), 
               mixture = 0.5) %>% 
  set_engine("glmnet") %>%
  set_mode("classification") 



en_wf <- 
  spec_wflow %>% 
  add_model(en_mod) 

en_res <- 
  en_wf %>% 
  tune_grid( 
            resamples = folds, 
            grid = lasso_reg_grid,
            control = ctrl_grid
            )




# ===================================



lda_mod <-
  discrim_linear(
    ) %>%
  set_engine("MASS") %>%
  set_mode("classification")

lda_wf <- spec_wflow %>%
  add_model(lda_mod)

lda_res <- 
  fit_resamples(
    lda_wf,
    resamples = folds,
    control = ctrl_res
  )

# ==================================


rf_mod <-
  rand_forest(
    mtry = floor(sqrt(ncol(train) - 1)), 
    trees = 500 
    ) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- spec_wflow %>%
  add_model(rf_mod)



rf_res <- 
  rf_wf %>% 
  fit_resamples(
    resamples = folds,
    control = ctrl_res
  )



## -----------------------------------------------------------------------------
model_st <- 
  stacks() %>%
  add_candidates(lda_res) %>%
  add_candidates(nnet_res) %>%
  add_candidates(rf_res) %>%
  add_candidates(en_res) %>%
  blend_predictions() %>%  
  fit_members()


## -----------------------------------------------------------------------------

select <- dplyr::select

ens_pred <-
  test_data %>%
  select(Response) %>%
  bind_cols(
    predict(
      model_st,
      test_data,
      type = "class",
      members = TRUE
      )
    )



ens_prob <-
  test_data %>%
  select(Response) %>%
  bind_cols(
    predict(
      model_st,
      test_data,
      type = "prob",
      members = TRUE
      )
    )



## -----------------------------------------------------------------------------
names(ens_pred) <- str_remove(names(ens_pred), ".pred_class_")
names(ens_pred) <- str_remove(names(ens_pred), "res_1_")
names(ens_pred) <- str_remove(names(ens_pred), "_1")
ens_pred <- ens_pred %>% rename(stack = .pred_class)



names(ens_prob) <- str_remove(names(ens_prob), ".pred_")
names(ens_prob) <- str_remove(names(ens_prob), "res_1_")
names(ens_prob) <- str_remove(names(ens_prob), "_1")
names(ens_prob)[2:4] <- str_c(names(ens_prob)[2:4], "_stack")


## -----------------------------------------------------------------------------
auc <- ens_prob %>% 
  mutate(id  = 1:nrow(ens_prob)) %>% 
  pivot_longer(-c(Response, id)) %>% 
  mutate(type = substr(name, 1, 3), 
         name = str_remove(name,"setosa_"), 
         name = str_remove(name,"versicolor_"),
         name = str_remove(name,"virginica_")) %>%
 pivot_wider(names_from = type, values_from = value) %>%
  group_by(name) %>% 
roc_auc(truth = Response,
  set:vir)

auc <- auc %>% select(name, .estimate) %>% pivot_wider(names_from = name, values_from = .estimate)

## -----------------------------------------------------------------------------
ens_prob <- ens_prob %>% 
  mutate(id  = 1:nrow(ens_prob)) %>% 
  pivot_longer(-c(Response, id)) %>% 
  mutate(type = substr(name, 1, 3), 
         name = str_remove(name,"setosa_"), 
         name = str_remove(name,"versicolor_"),
         name = str_remove(name,"virginica_")) %>%
  group_by(id, name) %>% 
  summarise(valuemax = max(value)) %>% 
  ungroup() %>%
  pivot_wider(id_cols = id, 
names_from = name, values_from = valuemax) %>%
  select(-id)


## -----------------------------------------------------------------------------
plot_ensemble(ens_pred %>% pull(Response), ens_pred %>% select(-Response))

## -----------------------------------------------------------------------------
plot_ensemble(ens_pred %>% pull(Response),
              ens_pred %>% select(-Response),
              incorrect = TRUE)

## -----------------------------------------------------------------------------
plot_ensemble(ens_pred %>% pull(Response),
              ens_pred %>% select(-Response),
              tibble_prob = ens_prob)


plot_ensemble(
  ens_pred %>% pull(Response),
  ens_pred %>% select(-Response),
  tibble_prob = ens_prob,
  incorrect = TRUE
)

## -----------------------------------------------------------------------------
auc <- auc[,
           names(ens_pred %>%
                   select(-Response))]

plot_ensemble(
  ens_pred %>%
    pull(Response),
  ens_pred %>% select(-Response),
  tibble_prob = ens_prob,
  order = auc
)


## -----------------------------------------------------------------------------


maj_vote <- apply(ens_pred %>%
                    select(-Response),
                  1,
                  function(x)
                    names(which.max(table(x))))


ens_pred <- ens_pred %>% mutate(maj_vote = as.factor(maj_vote))


plot_ensemble(ens_pred %>% pull(Response), ens_pred %>% select(-Response))

plot_ensemble(ens_pred %>% pull(Response),
              ens_pred %>% select(-Response),
              incorrect = TRUE)



## -----------------------------------------------------------------------------

prob_maj_vote <-
  apply(ens_pred %>% select(-Response), 1, function(x)
    max(table(x)) / length(x))

ens_prob <- ens_prob %>%
  mutate(maj_vote = prob_maj_vote)



plot_ensemble(ens_pred %>% pull(Response),
              ens_pred %>% select(-Response),
              tibble_prob = ens_prob)

plot_ensemble(
  ens_pred %>% pull(Response),
  ens_pred %>% select(-Response),
  tibble_prob = ens_prob,
  incorrect = TRUE
)



