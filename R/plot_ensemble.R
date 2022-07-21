#' Draws a plot for model predictions of ensembles of models. For classification the plot is a heatmap, for regression, scatterplot.
#'
#' @param truth The `y` variable. In regression this is `numeric` vector, in classification this is a `factor` vector.
#' @param tibble_pred A `data.frame` of predictions. Each column corresponds to a candidate model.
#' @param incorrect If `TRUE`, for observations that were correctly classified by all models, remove all but a single observation per class. Classification only.
#' @param tibble_prob If not `NULL`, a `data.frame` with same column names as `tibble_pred`. Applies transparency based on the predicted probability of the predicted class. Classification only.
#' @param order default ordering is by `accuracy` (classification) or `RMSE` (regression). Can submit any other ordering e.g. `AUC`, which should be a `data.frame` with same column names as `tibble_pred`.
#' @param facet whether to facet the plots by model (regression only).
#' @return a ggplot
#' @export
#' @import ggplot2
#' @importFrom magrittr %>%
#' @importFrom rlang is_installed
#' @examples
#' data(iris)
#' if (require("MASS")){
#' lda.model <- lda(Species~., data = iris)
#' lda.pred <- predict(lda.model)
#' }
#' if (require("ranger")){
#' ranger.model <- ranger(Species~., data = iris)
#' ranger.pred <- predict(ranger.model, iris)
#' }
#'
#' library(ensModelVis)
#'
#' plot_ensemble(iris$Species,
#' data.frame(LDA = lda.pred$class,
#' RF = ranger.pred$predictions))
#'
#' plot_ensemble(iris$Species,
#'  data.frame(LDA = lda.pred$class,
#'   RF = ranger.pred$predictions),
#'   incorrect= TRUE)
#'
#'if (require("ranger")){
#'ranger.model <- ranger(Species~., data = iris, probability = TRUE)
#'ranger.prob <- predict(ranger.model, iris)
#'}
#'
#'plot_ensemble(iris$Species,
#'   data.frame(LDA = lda.pred$class,
#'    RF = ranger.pred$predictions),
#'    tibble_prob = data.frame(LDA = apply(lda.pred$posterior, 1, max),
#'    RF = apply(ranger.prob$predictions, 1, max)))


plot_ensemble <- function(truth, tibble_pred, incorrect = FALSE,  tibble_prob = NULL, order = NULL, facet = FALSE){

  if (length(truth) != nrow(tibble_pred))
    stop("truth and predictions not same length.")
  if (!is.data.frame(tibble_pred))
    stop("Predictions are not a data frame.")
  if (is.factor(truth)) {
    if (sum(sapply(tibble_pred, is.factor)) != ncol(tibble_pred))
      stop("tibble_pred some columns not factors")
    if (!is.null(tibble_prob)) {
      if (!is.data.frame(tibble_prob))
        stop("Prediction probabilities are not a data frame.")
      if (!identical(sort(names(tibble_prob)),  sort(names(tibble_pred))))
        stop("tibble_prob does not match tibble_pred.")
      if (max(tibble_prob) > 1)
        stop("tibble_prob greater than 1.")
      if (min(tibble_prob) < 1 / length(levels(truth)))
        stop("tibble_prob less than 1/no levels.")
    }

    tib <- tibble_pred %>%
      dplyr::mutate(truth = truth)

    tib <- tib %>%
      tidyr::drop_na(truth)

    tib <- tib %>%
      dplyr::mutate(id = 1:nrow(tib))

    if (is.null(order)) {
      accuracy_vector <- tib %>%
        dplyr::summarise(dplyr::across(-.data$id,
                                       ~ accuracy_fun(truth, .x)))
    } else {
      if (!is.data.frame(order))
        stop("order is not a data frame")
      accuracy_vector <- order %>% dplyr::mutate(truth = 1)
      if (!identical(names(tib %>% dplyr::select(-.data$id)), names(accuracy_vector)))
        stop("order variable names not same as tibble_pred")
    }


    cols_order <- rank(-accuracy_vector,
                       ties.method = "last")

    tib <- tib[, c(names(sort(cols_order)), "id")]


    ord <-
      round(sort(as.numeric(accuracy_vector), decreasing = TRUE), 3)

    if (!is.null(tibble_prob)) {
      tib_prob <- tibble_prob %>%
        dplyr::mutate(truth = truth)

      tib_prob <- tib_prob %>%
        tidyr::drop_na(truth)

      tib_prob <- tib_prob %>%
        dplyr::mutate(id = 1:nrow(tib_prob))

      tib_prob <- tib_prob[, c(names(sort(cols_order)), "id")]


      # add prefix before joining:
      names(tib) <- paste0("x", names(tib))
      names(tib_prob) <- paste0("y", names(tib_prob))

      tib <- tib  %>%
        dplyr::rename(id = .data$xid)

      tib_prob <- tib_prob  %>%
        dplyr::rename(id = .data$yid)

      if (incorrect == TRUE) {
        dstinct <- !duplicated(tib %>%
                                 dplyr::select(-.data$id))
        tib <- tib[dstinct,]
        tib_prob <- tib_prob[dstinct,]
      }

      tib <- tib %>%
        dplyr::inner_join(tib_prob, by = "id") %>%
        dplyr::select(-.data$id) %>%
        dplyr::arrange(dplyr::across(dplyr::everything())) %>%
        dplyr::mutate(id = 1:nrow(tib))


      tib <- tib %>%
        dplyr::mutate(ytruth = rep(1, nrow(tib))) %>%
        tidyr::pivot_longer(-.data$id,
                            names_to = c(".value", "name"),
                            names_pattern = "(.)(.*)") %>%
        dplyr::mutate(
          name = as.factor(.data$name),
          name = forcats::fct_relevel(.data$name , names(sort(cols_order))),
          name = as.numeric(.data$name)
        ) %>%
        dplyr::rename(class = .data$x, prob = .data$y)


      p1 <- tib %>%
        ggplot(aes(x = .data$name, y = .data$id)) +
        geom_tile(aes(fill = class, alpha = .data$prob))

    }

    else{
      if (incorrect == TRUE) {
        dstinct <- !duplicated(tib %>% dplyr::select(-.data$id))
        tib <- tib[dstinct,]
      }
      tib <- tib %>%
        dplyr::select(-.data$id) %>%
        dplyr::arrange(dplyr::across(dplyr::everything())) %>% #
        dplyr::mutate(id = 1:nrow(tib)) %>%
        tidyr::pivot_longer(-.data$id) %>%
        dplyr::mutate(
          name = as.factor(.data$name),
          name = forcats::fct_relevel(.data$name , names(sort(cols_order))),
          name = as.numeric(.data$name)
        )


      p1 <- tib %>%  ggplot(aes(x = .data$name, y = .data$id)) +
        geom_tile(aes(fill = .data$value))

    }
    p1 <- p1 +
      theme_void() +
      scale_x_continuous(
        breaks = unique(tib$name),
        labels = names(sort(cols_order)),
        sec.axis = sec_axis(~ ., breaks = unique(tib$name), labels = ord)
      ) +
      theme(
        axis.title.x = element_blank(),
        plot.margin = margin(0.7, 0, 1.5, 0, "cm"),
        # axis.title.y=element_blank(),
        # axis.ticks.x=element_blank(),
        # axis.ticks.y=element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 0)
      ) +
      xlab("") +
      ylab("") +
      scale_fill_brewer(palette = "Set2")
  }else{
    if (!is.numeric(truth))
      stop("truth argument is neither factor not numeric")
    if (sum(sapply(tibble_pred , is.numeric)) != ncol(tibble_pred))
      stop("non-numeric estimates")


    tib <- tibble_pred %>%
      dplyr::mutate(truth = truth)

    tib <- tib %>%
      tidyr::drop_na(truth)

    if (facet == FALSE) {
      p1 <- tib %>%
        tidyr::pivot_longer(-truth) %>%
        dplyr::rename(algorithm = .data$name) %>%
        ggplot(aes(x = .data$truth, y = .data$value, col = .data$algorithm)) +
        geom_point() +
        ylab("Prediction") +
        xlab("Truth")
    } else{
      if (is.null(order)) {
        rmse_vector <- tib %>%
          dplyr::summarise(dplyr::across(dplyr::everything(),
                                         ~ rmse_fun(truth, .x)))
      } else {
        if (!is.data.frame(order))
          stop("order is not a data frame")
        rmse_vector <- order %>% dplyr::mutate(truth = 0)
        if (!identical(names(tib), names(rmse_vector)))
          stop("order variable names not same as tibble_pred")
      }


      cols_order <- rank(rmse_vector,
                         ties.method = "last")

      temp_rmse <-
        rmse_vector %>%
        tidyr::pivot_longer(dplyr::everything(), values_to = "RMSE") %>%
        dplyr::mutate(RMSE = round(.data$RMSE, 2))

      p1 <- tib %>%
        tidyr::pivot_longer(-truth) %>%
        dplyr::mutate(name = as.factor(.data$name),
               name = forcats::fct_relevel(.data$name,
                                           names(sort(cols_order))[-1])) %>%
        dplyr::left_join(temp_rmse) %>%
        ggplot(aes(x = .data$value, y = .data$truth)) +
        geom_point() +
        facet_wrap( ~ .data$RMSE + .data$name)
    }
  }
  p1
}
