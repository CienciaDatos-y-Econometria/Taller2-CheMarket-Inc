
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


##########?????

library(dplyr)
library(ggplot2)
library(forcats)
library(patchwork)
library(RColorBrewer)

# ==== 1) Bins y etiquetas ====
db2 <- db %>%
  mutate(
    # <<CAMBIA AQUÍ si tu variable no se llama 'sign_up'>>
    sign_up = sign_up,
    sessions_bin = cut(past_sessions,
                       breaks = c(-Inf, 4, 8, 14),
                       labels = c("0-4","5-8","9-14"),
                       right = TRUE),
    time_bin = cut(time_spent,
                   breaks = c(0, 6, 12, 18, 24, 30, 36, 42, 48, 55),
                   labels = c("0-6","7-12","13-18","19-24","25-30","31-36","37-42","43-48","49-55"),
                   include.lowest = TRUE, right = TRUE)
  )

# ==== 2) Paletas y orden EXACTOS ====
# Device: desktop / mobile / tablet
pal_device  <- c(desktop="#009E73", mobile="#0072B2", tablet="#E69F00")

# SO: osx / other / windows
pal_os      <- c(osx="#CC79A7", other="#0072B2", windows="#E69F00")

# Binarias: 0 / 1
pal_bin     <- c(`0`="#999999", `1`="#0072B2")

# Sesiones: 0-4 / 5-8 / 9-14
pal_sessions<- c("0-4"="#009E73", "5-8"="#0072B2", "9-14"="#E69F00")

# Tiempo en página (9 categorías)
pal_time <- setNames(
  c("#0072B2","#E69F00","#009E73","#D55E00","#CC79A7",
    "#F0E442","#56B4E9","#000000","#999999"),
  ord_time
)

ord_device  <- c("desktop","mobile","tablet")
ord_os      <- c("osx","other","windows")
ord_bin     <- c(0,1)
ord_sessions<- c("0-4","5-8","9-14")
ord_time    <- c("0-6","7-12","13-18","19-24","25-30","31-36","37-42","43-48","49-55")

# ==== 3) Función con estética del grupo ====
bar_mean <- function(data, x, title, xlab, order, colors, decimals=1) {
  x <- rlang::ensym(x)
  df <- data %>%
    filter(!is.na(!!x), !is.na(Revenue)) %>%
    group_by(val = !!x) %>%
    summarise(prom = mean(Revenue), .groups = "drop") %>%
    mutate(val = factor(val, levels = order))
  
  y_top <- max(df$prom) * 1.12
  
  ggplot(df, aes(x = val, y = prom, fill = val)) +
    geom_col(width = 0.78) +
    geom_text(aes(label = sprintf(paste0("%.",decimals,"f"), prom)),
              vjust = -0.45, fontface = "bold", size = 4.6) +
    scale_fill_manual(values = colors, guide = "none") +
    scale_y_continuous(limits = c(0, y_top), expand = expansion(mult = c(0, .02))) +
    labs(title = title, x = xlab, y = "Gasto promedio") +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.title.x = element_text(margin = margin(t = 6)))
}

# ==== 4) Gráficos individuales ====
g_reg <- bar_mean(db2, sign_up, "Revenue según registro", "Registro", ord_bin, pal_bin) # <<y AQUÍ si cambias el nombre>>
g_ret <- bar_mean(db2, is_returning_user, "Revenue según usuario recurrente", "Usuario Recurrente", ord_bin, pal_bin)
g_os  <- bar_mean(db2, os_type, "Revenue según sistema operativo", "Sistema Operativo", ord_os, pal_os)
g_dev <- bar_mean(db2, device_type, "Revenue según dispositivo", "Dispositivo", ord_device, pal_device)
g_ses <- bar_mean(db2, sessions_bin, "Revenue según número de sesiones", "Número de sesiones", ord_sessions, pal_sessions)
g_tim <- bar_mean(db2, time_bin, "Revenue según tiempo en página", "Tiempo en página", ord_time, pal_time)

# ==== 5) Panel 3x2 como el de la imagen ====
panel_prom <- (g_reg | g_ret) / (g_os | g_dev) / (g_ses | g_tim)
panel_prom

# (Opcional) guardar en buena resolución
# ggsave("panel_promedios_like_compas.png", panel_prom, width = 12, height = 14, dpi = 300)





















































##########?????




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

