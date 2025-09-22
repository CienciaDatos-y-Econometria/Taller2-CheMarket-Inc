# -----------------------------------------------------
# PART 1
# Este script implementa árboles de regresión intentando
# mejorar la predicción de ingreso para CheMarket
# 0) Crear datos de entrenamiento y validación 
# 1) Modelos estimados en Taller 1, y crear train y test
# 2) Estimar mejores modelos con Lasso y Ridge
# 3) Estimar modelos con árboles de regresión
# 3.1) Mejor modelo según profundidad
# 3.2) Mejor modelo según observaciones en nodos
# 4) Graficar mejores árboles según ambos criterios
# 5) Comparar todos los modelos juntos 
# -----------------------------------------------------

# -----------------------------------------------------
# 0) Crear datos de entrenamiento y validación
# -----------------------------------------------------

# Cargar datos y crear variable log_revenue
db <- parte_a  # Usar datos de la parte A
db <- db %>% mutate(log_revenue = log(Revenue))  # Crear variable log_revenue
db <- db %>% mutate(sqrt_time_spent = sqrt(time_spent)) # Variable sqrt(time_spent)

# Fijamos la semilla
set.seed(2025)

# Crear trainning data (30% of observations)
smp_size <- floor(0.3 * nrow(db))

# Creamos la columna de validación en la db para separar
validacion_ids <- sample(seq_len(nrow(db)), size = smp_size)
db$validacion <- 0
db$validacion[validacion_ids] <- 1

# test and trainning sets
data_entreno <- db %>% filter(validacion == 0)
data_validacion <- db %>% filter(validacion == 1)

# -----------------------------------------------------
# 1) Modelos estimados en Taller 1, y crear train y test
# -----------------------------------------------------

# Modelos Taller 1
model_basico <- paste0("log_revenue ~ ", "sign_up")

model_todas <- paste0(model_basico, " + ",
                      "sqrt_time_spent + past_sessions + device_type + 
                      is_returning_user + os_type"
)
                      
model_interacciones <- paste0(model_todas, " + ",
                      "sign_up:past_sessions + sign_up:os_type + 
                      sign_up:device_type + sign_up:is_returning_user"
)

model_cuadratico <- paste0(model_interacciones, " + ",
                      "sqrt_time_spent*sqrt_time_spent" #TODO: para qué, si ya sacamos la raíz?
)
models_t1 <- list(
  model_basico,
  model_todas,
  model_interacciones,
  model_cuadratico
)

predictor <- function(regresors){
  fmla <- formula(regresors)
  model <- lm(fmla, data = data_entreno)
  prediction_test <- predict(model, newdata = data_validacion)
  mse <- with(data_validacion, mean((log_revenue - prediction_test)^2))
  return(mse)
}

# Sacar MSE para los modelos
lapply(models_t1, predictor)


# -----------------------------------------------------
# 2) Estimar mejores modelos con Lasso y Ridge
# -----------------------------------------------------

# Poner los datos en matrices para caret
X <- model.matrix(formula(model_cuadratico), data_entreno)[, -1]  # Eliminar intercepto
Y <- data_entreno$log_revenue  # Usar log_revenue para consistencia

# Semilla para reproducibilidad
set.seed(2025)

# Lasso
modelo_lasso <- cv.glmnet(
  x=X,
  y=Y,
  alpha = 1, # Lasso
  nfolds = 10, #Val. cruzada 10-fold
  type.measure = "mse" 
)

# Ridge
modelo_ridge <- cv.glmnet(
  x=X,
  y=Y,
  alpha = 0, # Ridge
  nfolds = 10, # Val. cruzada 10-fold
  type.measure = "mse"
)

# Poner datos de validación en matrices
X_test <- model.matrix(formula(model_cuadratico), data_validacion)[, -1]
y_test <- data_validacion$log_revenue  # Usar log_revenue para consistencia


# Predicciones con lambda.min (LASSO)
pred_lasso_min <- predict(modelo_lasso, newx = X_test, s = "lambda.min")
pred_lasso_min <- as.numeric(pred_lasso_min)  # Convertir a vector

# Predicciones con lambda.1se (LASSO)
pred_lasso_1se <- predict(modelo_lasso, newx = X_test, s = "lambda.1se")
pred_lasso_1se <- as.numeric(pred_lasso_1se)

# Predicciones con lambda.min (RIDGE)
pred_ridge_min <- predict(modelo_ridge, newx = X_test, s = "lambda.min")
pred_ridge_min <- as.numeric(pred_ridge_min)

# Predicciones con lambda.1se (RIDGE)
pred_ridge_1se <- predict(modelo_ridge, newx = X_test, s = "lambda.1se")
pred_ridge_1se <- as.numeric(pred_ridge_1se)

# Función para calcular múltiples métricas
calculate_metrics <- function(y_true, y_pred, model_name) {
  residuals <- y_true - y_pred
  
  # Calcular métricas correctas
  mse <- mean(residuals^2)                    # Error cuadrático medio
  rmse <- sqrt(mse)                           # Raíz del error cuadrático medio  
  
  # R-cuadrado
  ss_res <- sum(residuals^2)                  # Suma de cuadrados residuales
  ss_tot <- sum((y_true - mean(y_true))^2)    # Suma de cuadrados totales
  r_squared <- 1 - (ss_res / ss_tot)          # R²
  
  metrics <- tibble(
    modelo = model_name,
    MSE = mse,               
    RMSE = rmse,
    R_squared = r_squared
  )
  
  return(metrics)
}

# Calcular métricas para todos los modelos regularizados
metrics_lasso_min <- calculate_metrics(y_test, pred_lasso_min, "Lasso (lambda.min)")
metrics_lasso_1se <- calculate_metrics(y_test, pred_lasso_1se, "Lasso (lambda.1se)")
metrics_ridge_min <- calculate_metrics(y_test, pred_ridge_min, "Ridge (lambda.min)")
metrics_ridge_1se <- calculate_metrics(y_test, pred_ridge_1se, "Ridge (lambda.1se)")

# TODO: esto va al final, cuando se evalúen todos los modelos entre si
# Combinar todas las métricas en una tabla
metricas_regularizacion <- bind_rows(
  metrics_lasso_min,
  metrics_lasso_1se,
  metrics_ridge_min,
  metrics_ridge_1se
)

# Mostrar resultados
print("=== COMPARACIÓN DE MODELOS REGULARIZADOS ===")
print(metricas_regularizacion)

# Mostrar los lambdas óptimos encontrados
cat("\n=== VALORES LAMBDA ÓPTIMOS ===\n")
cat("LASSO:\n")
cat("  lambda.min =", modelo_lasso$lambda.min, "\n")
cat("  lambda.1se =", modelo_lasso$lambda.1se, "\n")
cat("RIDGE:\n")
cat("  lambda.min =", modelo_ridge$lambda.min, "\n")
cat("  lambda.1se =", modelo_ridge$lambda.1se, "\n")


# -----------------------------------------------------
# 3.1) Mejor modelo según profundidad (maxdepth)
# -----------------------------------------------------

# Control de entrenamiento con validación cruzada
fitControl <- trainControl(method = "cv", number = 10)

# Semilla replicabilidad
set.seed(2025)

# Modelo por PROFUNDIDAD (rpart2)
tree_profundidad <- train(
  formula(model_cuadratico),  # Usamos el modelo más complejo
  data = data_entreno,        # Usar datos de entrenamiento
  method = "rpart2",          # método profundidad
  trControl = fitControl,
  tuneGrid = expand.grid(maxdepth = seq(1, 10, 1))  
) # Probar profundidades de 1 a 10; hay máximo 10 variables

# Ver los mejores hiperparámetros
print("=== MEJOR MODELO POR PROFUNDIDAD ===")
print(tree_profundidad$bestTune)
print(tree_profundidad$results)


# -----------------------------------------------------
# 3.2) Mejor modelo según observaciones en nodos (minsplit)
# -----------------------------------------------------

# Modelo por OBSERVACIONES MÍNIMAS (rpart)
set.seed(2025)
tree_observaciones <- train(
  formula(model_cuadratico),  # Usamos el modelo más complejo
  data = data_entreno,        # Usar datos de entrenamiento
  method = "rpart",           # método por observaciones mínimas
  trControl = fitControl,
  tuneGrid = expand.grid(cp = seq(0.00001, 0.0001, 0.00005))  
)  # Probar complejidades de 0.001 a 0.1 en pasos de 0.005
# se probó seq(0.001, 0.1, 0.005) y el mejor fue 0.001
# también seq(0.0001, 0.01, 0.0005) y también dió el mínimo

# Ver los mejores hiperparámetros
print("=== MEJOR MODELO POR COMPLEJIDAD ===")
print(tree_observaciones$bestTune)
print(tree_observaciones$results)


# -----------------------------------------------------
# 4) Predicciones y evaluación de árboles
# -----------------------------------------------------

# Predicciones en conjunto de validación
pred_tree_profundidad <- predict(tree_profundidad, newdata = data_validacion)
pred_tree_observaciones <- predict(tree_observaciones, newdata = data_validacion)

# Calcular métricas para árboles
metrics_tree_prof <- calculate_metrics(y_test, pred_tree_profundidad, "Árbol (profundidad)")
metrics_tree_obs <- calculate_metrics(y_test, pred_tree_observaciones, "Árbol (complejidad)")

# Combinar métricas de árboles
metricas_arboles <- bind_rows(
  metrics_tree_prof,
  metrics_tree_obs
)

print("=== COMPARACIÓN DE MODELOS DE ÁRBOLES ===")
print(metricas_arboles)


# -----------------------------------------------------
# 5) Diagramar los mejores árboles
# -----------------------------------------------------

# Graficar árbol por profundidad
library(rpart.plot)
print("=== ÁRBOL OPTIMIZADO POR PROFUNDIDAD ===")
rpart.plot(tree_profundidad$finalModel, 
           main = "Árbol optimizado por profundidad",
           box.palette = "RdYlGn",
           shadow.col = "gray",
           nn = TRUE)

# Graficar árbol por observaciones
print("=== ÁRBOL OPTIMIZADO POR COMPLEJIDAD ===")
rpart.plot(tree_observaciones$finalModel, 
           main = "Árbol optimizado por complejidad", 
           box.palette = "RdYlGn",
           shadow.col = "gray",
           nn = TRUE)

# Caret


# -----------------------------------------------------
# 6) Comparación final de todos los modelos
# -----------------------------------------------------

# Combinar todas las métricas (regularización + árboles)
comparacion_final <- bind_rows(
  metricas_regularizacion,
  metricas_arboles
)

print("=== COMPARACIÓN FINAL DE TODOS LOS MODELOS ===")
print(comparacion_final)

# Encontrar el mejor modelo por cada métrica
mejor_MSE <- comparacion_final[which.min(comparacion_final$MSE), ]
mejor_RMSE <- comparacion_final[which.min(comparacion_final$RMSE), ]
mejor_MAE <- comparacion_final[which.min(comparacion_final$MAE), ]
mejor_R2 <- comparacion_final[which.max(comparacion_final$R_squared), ]

cat("\n=== MEJORES MODELOS POR MÉTRICA ===\n")
cat("Mejor MSE:", mejor_MSE$modelo, "con MSE =", mejor_MSE$MSE, "\n")
cat("Mejor RMSE:", mejor_RMSE$modelo, "con RMSE =", mejor_RMSE$RMSE, "\n") 
cat("Mejor MAE:", mejor_MAE$modelo, "con MAE =", mejor_MAE$MAE, "\n")
cat("Mejor R²:", mejor_R2$modelo, "con R² =", mejor_R2$R_squared, "\n")


