
# -----------------------------------------------------
# PART 1
# Exploratory and predictive analysis from observed data
# 0) Libraries and data
# 1) Description of users general behavior
# 2) Estimate effect of "sign_up" into "revenue"
# 3) Logistic regression of variables into sign_up
# r) Evaluating model's predictive capacity
# -----------------------------------------------------

# -----------------------------------------------------
# 0) Clean variables, libraries and data
# -----------------------------------------------------

# Libraries
require(pacman)
p_load(dplyr, tidyr, tidyverse, readr, ggplot2, corrplot)

# Change data name
db <- parte_a

# See and keep variables
vars <- colnames(db)

# Get categoric and non-categoric variables
categoric_vars <- sapply(db, is.factor)
non_categoric_vars <- !categoric_vars

# -----------------------------------------------------
# 1) Description of users general behaviour
# -----------------------------------------------------
# Summary statistics
summary(db)
# TODO exportar

# Qué graficar con cada tipo
dens_vars <- c("time_spent", "Revenue")
hist_vars <- "past_sessions"

# Función de densidad (bonita)
plot_density <- function(var_name) {
  ggplot(db, aes_string(x = var_name)) +
    geom_density(na.rm = TRUE, linewidth = 0.8, fill = "#6aa6f8", alpha = 0.35) +
    labs(title = paste("Densidad de", var_name),
         x = var_name, y = "Densidad") +
    theme_minimal(base_size = 12)
}

# Histograma con binwidth adaptativo (Freedman–Diaconis con fallback)
plot_hist <- function(var_name) {
  x <- db[[var_name]]
  b0 <- floor(min(x, na.rm = TRUE)) - 0.5  # centra cada barra en los enteros
  ggplot(db, aes_string(x = var_name)) +
    geom_histogram(
      binwidth = 1, boundary = b0, closed = "left",
      na.rm = TRUE, linewidth = 0.3, fill = "#3a5e8c", alpha = 0.85
    ) +
    scale_x_continuous(
      breaks = seq(floor(min(x, na.rm = TRUE)), ceiling(max(x, na.rm = TRUE)), by = 1),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = paste("Histograma de", var_name),
         x = var_name, y = "Frecuencia") +
    theme_minimal(base_size = 12)
}

# Imprimir las densidades (time_spent, Revenue)
invisible(lapply(dens_vars, function(v) print(plot_density(v))))
# Imprimir el histograma (past_sessions)
invisible(lapply(hist_vars, function(v) print(plot_hist(v))))

# Log-Revenue y sqrt-time (vemos que si mejoran y lo cambiamos en la db)
#TODO: exportar log-rev para anexo
plot(density(log(db$Revenue)), main = "Density of log-Revenue")
plot(density(sqrt(db$time_spent)), main = "Density sqrt-time")

db <- db %>% mutate(log_revenue = log(Revenue))
db <- db %>% mutate(sqrt_time_spent = sqrt(time_spent))


### Ya verificamos las variables. Seleccionemos las dependientes del modelo


# See pairs (log(revenue) vs continuas) (correlations)
# TODO exportar
str <- paste0("~", paste(names(db)[non_categoric_vars], collapse = "+"))
pairs(db[, names(db)[non_categoric_vars]])


# Box-wiskers (Para no categóricas) 
#TODO no sé si esto lo necesito realmente
# TODO exportar
box_f <-  function(var_name) {
  boxplot(db[[var_name]], main = paste("Boxplot of", var_name))
}
lapply(names(db)[non_categoric_vars], box_f)

# Boxplots categóricas vs revenue
# Todo: gráficas bien, pero imprime un montón de números 
# TODO exportar
box_f_cat_rev <-  function(var_name) {
  boxplot(db$log_revenue ~ db[[var_name]], main = paste("Boxplot of log(Revenue) by", var_name))
}
lapply(names(db)[categoric_vars], box_f_cat_rev)




#
# Función: Boxplot + diferencia de medias
box_f_cat_rev <- function(var_name) {
  # Boxplot
  boxplot(db$log_revenue ~ db[[var_name]], 
          main = paste("Boxplot of log(Revenue) by", var_name))
  
  # Diferencia de medias
  means <- tapply(db$log_revenue, db[[var_name]], mean, na.rm = TRUE)
  diffs <- combn(means, 2, FUN = function(x) diff(rev(x))) # diferencias entre categorías

  # Graficar resultados con ggplot2 y ponlo bonito
  df <- data.frame(Category = names(means), Mean = means)
  ggplot(df, aes(x = Category, y = Mean)) +
    geom_bar(stat = "identity") +
    labs(title = paste("Mean of log(Revenue) by", var_name))
}

# Aplicar a todas las categóricas
lapply(names(db)[categoric_vars], box_f_cat_rev)




# Función: medias + IC95% y gráfico con barras + errorbars
plot_mean_diff <- function(var_name) {
  
  # Calcular medias e IC95%
  summary_df <- db %>%
    group_by(!!sym(var_name)) %>%
    summarise(
      mean_rev = mean(log_revenue, na.rm = TRUE),
      sd_rev   = sd(log_revenue, na.rm = TRUE),
      n        = n(),
      .groups = "drop"
    ) %>%
    mutate(
      se = sd_rev / sqrt(n),
      ci_low = mean_rev - 1.96 * se,
      ci_high = mean_rev + 1.96 * se
    )
  
  # Gráfico: barras + error bars en negro
  ggplot(summary_df, aes(x = !!sym(var_name), y = mean_rev, fill = !!sym(var_name))) +
    geom_col(width = 0.6, alpha = 0.8) +   # Barras
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), 
                  width = 0.2, color = "black", size = 0.8) +  # Intervalos en negro
    labs(
      title = paste("Media de log(Revenue) por", var_name),
      x = var_name,
      y = "Mean log(Revenue)"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
}

# Aplicar la función a todas las variables categóricas
plots <- lapply(names(db)[categoric_vars], plot_mean_diff)

# Mostrar el primero como ejemplo
plots[[1]]
plots[[2]]
plots[[3]]
plots[[4]]
# TODO: exportar


# Categorics vs sign_up
# TODO exportar
cat_vs_cat <- function(var1, var2) {
  ggplot(db, aes_string(x = var1, fill = var2)) +
    geom_bar(position = "dodge") +
    labs(title = paste("Count of", var2, "by", var1))
}

lapply(names(db)[categoric_vars], function(var1) {
  cat_vs_cat(var1, "sign_up")
})



# -----------------------------------------------------
# 2) Estimate effect of "sign_up" into "revenue"
# -----------------------------------------------------

#TODO: si hay tiempo, puedo regresar usando lapply
# Simple linear regression model
model <- lm(log_revenue ~ sign_up, data = db)
# TODO exportar
summary(model)

# Controlling by variables related to Revenue
model_controlled <- lm(log(Revenue) ~ sign_up + sqrt_time_spent + past_sessions + 
                         device_type + is_returning_user + os_type, data = db)
# TODO exportar
summary(model_controlled)

# Controlar por interacciones entre vars fuertemente relacionadas a Revenue
# TODO: Rev vs time y vs past_Sessions sube al principio y luego baja
# Con device_type, os_type y sign_up no es tan claro; con returning_user si


# Modelo final (quitando aquellas anteriores no significativas)



# -----------------------------------------------------
# 3) Logistic regression of variables into sign_up
# -----------------------------------------------------

# Quitamos revenue del set de variables explicativas
# Falta justificar past_sessions, sqrt_time_spent
vars <- setdiff(names(db), c("sign_up", "Revenue", "log_revenue", "time_spent", "validacion"))

# Fórmula dinámica
fml <- as.formula(paste("sign_up ~", paste(vars, collapse = " + ")))

# Ajustamos el modelo logístico
logit_model <- glm(fml, data = db, family = binomial(link = "logit"))

# Resumen del modelo
summary(logit_model)

# Odds ratios en vez de coeficientes logit
exp(coef(logit_model))


# -----------------------------------------------------
# 4) Evaluating model's predictive capacity
# -----------------------------------------------------

# Fijamos la semilla
set.seed(2025)

# Crear trainning data (30% of observations)
smp_size <- floor(0.3 * nrow(db))

# Creamos la columna de validación en la db para separar
validacion_ids <- sample(seq_len(nrow(db)), size = smp_size)
db$validacion <- 0
db$validacion[validacion_ids] <- 1

# test and trainning sets
data_test <- db %>% filter(validacion == 1)
data_train <- db %>% filter(validacion == 0)

# Poner variables a los modelos para data_train
#TODO

models <- list(
  "sign_up", # Básico
  "sign_up + sqrt_time_spent + past_sessions + device_type + is_returning_user + os_type", # Todas variables
  "sign_up + sqrt_time_spent + past_sessions + device_type + is_returning_user + os_type +
   sign_up:past_sessions + sign_up:os_type + sign_up:device_type + sign_up:is_returning_user", # Interacciones
  "sign_up + sqrt_time_spent + past_sessions + device_type + is_returning_user + os_type +
   sign_up:past_sessions + sign_up:os_type + sign_up:device_type + sign_up:is_returning_user + # Interacciones
   sqrt_time_spent*sqrt_time_spent"
)



predictor<-function(regresors){
  fmla<- formula(paste0("log_revenue ~ ",regresors))
  model <- lm(fmla,data = data_train)
  prediction_test <- predict(model, newdata = data_test)
  mse<-with(data_test,mean((log_revenue-prediction_test)^2))
  return(mse)
}


# Sacar MSE para los 
lapply(models, predictor)

