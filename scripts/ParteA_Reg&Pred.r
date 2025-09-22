# -----------------------------------------------------
# PARTE 1
# Este script implementa árboles de regresión intentando
# mejorar la predicción de ingreso para CheMarket
# 0) Crear datos de entrenamiento y validación 
# 1) Modelos estimados en Taller 1, y crear train y test
# 2) Estimar mejores modelos con Lasso y Ridge
# 3) Estimar modelos con árboles de regresión
# 3.1) Mejor modelo según profundidad
# 3.2) Mejor modelo según observaciones en nodos
# 4) Predicciones y evaluación de árboles
# 5) Diagramar los mejores árboles
# 6) Comparar todos los modelos juntos
# 7) Crear tabla mejorada con hiperparámetros
# 8) Exportar tablas a diferentes formatos
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

# Lista de modelos
models_t1 <- list(
  model_basico,
  model_todas,
  model_interacciones
)

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

# Función para entrenar y predecir, y calcular métricas regresión lineal
predictor <- function(regresors, model_name){
  fmla <- formula(regresors)
  model <- lm(fmla, data = data_entreno)
  prediction_test <- predict(model, newdata = data_validacion)
  
  # Usar la función calculate_metrics para obtener todas las métricas
  metrics <- calculate_metrics(data_validacion$log_revenue, prediction_test, model_name)
  return(metrics)
}

# Aplicar a todos los modelos del Taller 1
nombres_modelos <- c("LM Básico", "LM Todas", "LM Interacciones", "LM Cuadrático")

# Calcular métricas para todos los modelos de regresión lineal
metricas_lm <- data.frame()
for(i in 1:length(models_t1)) {
  metrics <- predictor(models_t1[[i]], nombres_modelos[i])
  metricas_lm <- bind_rows(metricas_lm, metrics)
}

print("=== RESULTADOS MODELOS DE REGRESIÓN LINEAL ===")
print(metricas_lm)

# Encontrar el mejor modelo de regresión lineal
mejor_lm <- metricas_lm[which.min(metricas_lm$MSE), ]
cat("Mejor modelo de regresión lineal:", mejor_lm$modelo, "con MSE =", mejor_lm$MSE, "\n")


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
# 5) Diagramar los mejores árboles con estilo mejorado
# -----------------------------------------------------

# Función personalizada para mostrar valores en los nodos
nodo_personalizado <- function(x, labs, digits, varlen) {
  # Verificar si x es numérico y tiene la estructura correcta
  if (is.numeric(x) && length(x) > 0) {
    # Usar el valor promedio predicho en el nodo
    valor <- x[1]  # Tomar el primer valor si es un vector
    return(paste("Log(Revenue) \n", format(round(valor, 4), nsmall=4, big.mark=",")))
  } else {
    # Si no es numérico, usar las etiquetas por defecto
    return(labs)
  }
}

print("=== ÁRBOL OPTIMIZADO POR PROFUNDIDAD ===")

# Guardar gráfico del árbol por profundidad
png("views/arbol_profundidad.png", width = 1200, height = 800, res = 150)
prp(tree_profundidad$finalModel, 
    under = TRUE,              # Muestra información adicional debajo del nodo
    branch.lty = 2,            # Estilo de línea punteada para las ramas
    yesno = 2,                 # Muestra "sí/no" en las bifurcaciones
    faclen = 0,                # Muestra etiquetas completas de factores
    varlen = 15,               # Longitud máxima del nombre de la variable
    tweak = 1.2,               # Ajusta el tamaño del texto
    clip.facs = TRUE,          # Recorta niveles largos de factores
    box.palette = "Greens",    # Paleta de colores para las cajas
    compress = TRUE,           # Comprime el árbol verticalmente
    ycompress = TRUE,          # Comprime también el eje y
    main = "Árbol Optimizado por Profundidad",
    digits = 4                 # Mostrar 4 decimales en los valores
)
dev.off()

print("=== ÁRBOL OPTIMIZADO POR COMPLEJIDAD ===")

# Guardar gráfico del árbol por complejidad  
png("views/arbol_complejidad.png", width = 1200, height = 800, res = 150)
prp(tree_observaciones$finalModel, 
    under = TRUE,              # Muestra información adicional debajo del nodo
    branch.lty = 2,            # Estilo de línea punteada para las ramas
    yesno = 2,                 # Muestra "sí/no" en las bifurcaciones
    faclen = 0,                # Muestra etiquetas completas de factores
    varlen = 15,               # Longitud máxima del nombre de la variable
    tweak = 1.2,               # Ajusta el tamaño del texto
    clip.facs = TRUE,          # Recorta niveles largos de factores
    box.palette = "Blues",     # Paleta de colores diferente para distinguir
    compress = TRUE,           # Comprime el árbol verticalmente
    ycompress = TRUE,          # Comprime también el eje y
    main = "Árbol Optimizado por Complejidad (cp)",
    digits = 4                 # Mostrar 4 decimales en los valores
)
dev.off()

# También mostrar en pantalla (versión simplificada)
par(mfrow = c(1, 2))  # Dos gráficos lado a lado

# Árbol profundidad (pantalla)
prp(tree_profundidad$finalModel, 
    box.palette = "Greens",
    main = "Árbol por Profundidad",
    tweak = 1.0,
    digits = 4)

# Árbol complejidad (pantalla)  
prp(tree_observaciones$finalModel,
    box.palette = "Blues", 
    main = "Árbol por Complejidad",
    tweak = 1.0,
    digits = 4)

par(mfrow = c(1, 1))  # Restaurar layout original

cat("\n✅ Gráficos de árboles guardados en:\n")
cat("   - views/arbol_profundidad.png\n") 
cat("   - views/arbol_complejidad.png\n")

# Caret


# -----------------------------------------------------
# 6) Comparación final de todos los modelos
# -----------------------------------------------------

# Combinar todas las métricas (regresión lineal + regularización + árboles)
comparacion_final <- bind_rows(
  metricas_lm,                # Modelos de regresión lineal
  metricas_regularizacion,    # Modelos Lasso y Ridge
  metricas_arboles           # Modelos de árboles
)

print("=== COMPARACIÓN FINAL DE TODOS LOS MODELOS ===")
print(comparacion_final)

# Encontrar el mejor modelo por cada métrica
mejor_MSE <- comparacion_final[which.min(comparacion_final$MSE), ]
mejor_RMSE <- comparacion_final[which.min(comparacion_final$RMSE), ]
mejor_R2 <- comparacion_final[which.max(comparacion_final$R_squared), ]

cat("\n=== MEJORES MODELOS POR MÉTRICA ===\n")
cat("Mejor MSE:", mejor_MSE$modelo, "con MSE =", mejor_MSE$MSE, "\n")
cat("Mejor RMSE:", mejor_RMSE$modelo, "con RMSE =", mejor_RMSE$RMSE, "\n") 
cat("Mejor R²:", mejor_R2$modelo, "con R² =", mejor_R2$R_squared, "\n")


# -----------------------------------------------------
# 7) Crear tabla mejorada con hiperparámetros
# -----------------------------------------------------

# Función para crear tabla mejorada
crear_tabla_mejorada <- function() {
  
  # Crear una copia de la comparación final
  tabla_mejorada <- comparacion_final
  
  # Agregar columna de hiperparámetros
  tabla_mejorada$Hiperparametro <- ""
  tabla_mejorada$Valor_Hiperparametro <- ""
  
  # Agregar información de hiperparámetros
  for(i in 1:nrow(tabla_mejorada)) {
    modelo <- tabla_mejorada$modelo[i]
    
    if(grepl("Lasso.*lambda.min", modelo)) {
      tabla_mejorada$modelo[i] <- "Lasso (λ min)"
      tabla_mejorada$Hiperparametro[i] <- "λ"
      tabla_mejorada$Valor_Hiperparametro[i] <- sprintf("%.6f", modelo_lasso$lambda.min)
    } else if(grepl("Lasso.*lambda.1se", modelo)) {
      tabla_mejorada$modelo[i] <- "Lasso (λ 1se)"
      tabla_mejorada$Hiperparametro[i] <- "λ"
      tabla_mejorada$Valor_Hiperparametro[i] <- sprintf("%.6f", modelo_lasso$lambda.1se)
    } else if(grepl("Ridge.*lambda.min", modelo)) {
      tabla_mejorada$modelo[i] <- "Ridge (λ min)"
      tabla_mejorada$Hiperparametro[i] <- "λ"
      tabla_mejorada$Valor_Hiperparametro[i] <- sprintf("%.6f", modelo_ridge$lambda.min)
    } else if(grepl("Ridge.*lambda.1se", modelo)) {
      tabla_mejorada$modelo[i] <- "Ridge (λ 1se)"
      tabla_mejorada$Hiperparametro[i] <- "λ"
      tabla_mejorada$Valor_Hiperparametro[i] <- sprintf("%.6f", modelo_ridge$lambda.1se)
    } else if(grepl("profundidad", modelo)) {
      tabla_mejorada$modelo[i] <- "Árbol (Profundidad)"
      tabla_mejorada$Hiperparametro[i] <- "maxdepth"
      tabla_mejorada$Valor_Hiperparametro[i] <- as.character(tree_profundidad$bestTune$maxdepth)
    } else if(grepl("minsplit|complejidad", modelo)) {
      tabla_mejorada$modelo[i] <- "Árbol (Complejidad)"
      tabla_mejorada$Hiperparametro[i] <- "cp"
      tabla_mejorada$Valor_Hiperparametro[i] <- sprintf("%.6f", tree_observaciones$bestTune$cp)
    } else {
      # Modelos de regresión lineal sin hiperparámetros
      tabla_mejorada$Hiperparametro[i] <- "-"
      tabla_mejorada$Valor_Hiperparametro[i] <- "-"
    }
  }
  
  # Reorganizar columnas y ordenar según el orden deseado
  tabla_mejorada <- tabla_mejorada %>%
    select(modelo, Hiperparametro, Valor_Hiperparametro, MSE, RMSE, R_squared)
  
  # Crear un orden específico: Regresión -> Ridge -> Lasso -> Árboles
  orden_deseado <- c("LM Básico", "LM Todas", "LM Interacciones", "LM Cuadrático",
                     "Ridge (λ min)", "Ridge (λ 1se)", 
                     "Lasso (λ min)", "Lasso (λ 1se)",
                     "Árbol (Profundidad)", "Árbol (Complejidad)")
  
  # Reordenar la tabla según el orden deseado
  tabla_mejorada$modelo <- factor(tabla_mejorada$modelo, levels = orden_deseado)
  tabla_mejorada <- tabla_mejorada %>% arrange(modelo)
  tabla_mejorada$modelo <- as.character(tabla_mejorada$modelo)
  
  return(tabla_mejorada)
}

# Crear la tabla mejorada
tabla_final <- crear_tabla_mejorada()

print("=== TABLA FINAL CON HIPERPARÁMETROS ===")
print(tabla_final)

# Crear versión "bonita" para presentación
crear_tabla_bonita <- function(tabla) {
  tabla_bonita <- tabla
  
  # Formatear números con menos decimales para presentación
  tabla_bonita$MSE <- sprintf("%.4f", tabla_bonita$MSE)
  tabla_bonita$RMSE <- sprintf("%.4f", tabla_bonita$RMSE) 
  tabla_bonita$R_squared <- sprintf("%.4f", tabla_bonita$R_squared)
  
  # Para modelos con hiperparámetros, agregar el valor debajo del RMSE
  for(i in 1:nrow(tabla_bonita)) {
    if(tabla_bonita$Valor_Hiperparametro[i] != "-") {
      # Formato: RMSE \n (hiperparámetro = valor)
      tabla_bonita$RMSE[i] <- paste0(tabla_bonita$RMSE[i], "\n(", 
                                     tabla_bonita$Hiperparametro[i], " = ", 
                                     tabla_bonita$Valor_Hiperparametro[i], ")")
    }
  }
  
  # Eliminar columnas de hiperparámetros ya que están en RMSE
  tabla_bonita <- tabla_bonita %>% 
    select(modelo, MSE, RMSE, R_squared)
  
  colnames(tabla_bonita) <- c("Modelo", "MSE", "RMSE", "R²")
  
  return(tabla_bonita)
}

tabla_bonita <- crear_tabla_bonita(tabla_final)

print("\n=== TABLA BONITA PARA PRESENTACIÓN ===")
print(tabla_bonita)


# -----------------------------------------------------
# 8) Exportar tablas a diferentes formatos
# -----------------------------------------------------

# Crear carpeta views si no existe
if (!dir.exists("views")) {
  dir.create("views")
}

# Exportar tabla completa a CSV
write.csv(tabla_final, "views/resultados_modelos_completos.csv", row.names = FALSE)
cat("\n✅ Tabla completa exportada a: views/resultados_modelos_completos.csv\n")

# Exportar tabla bonita a CSV
write.csv(tabla_bonita, "views/resultados_modelos_presentacion.csv", row.names = FALSE)
cat("✅ Tabla para presentación exportada a: views/resultados_modelos_presentacion.csv\n")

# Crear tabla para LaTeX (versión académica)
library(knitr)

# Tabla completa para apéndice
tabla_latex_completa <- kable(tabla_final, 
                             format = "latex", 
                             digits = 4,
                             col.names = c("Modelo", "Hiperparámetro", "Valor", "MSE", "RMSE", "R²"),
                             caption = "Comparación completa de modelos con hiperparámetros")

# Tabla bonita para cuerpo del documento
tabla_latex_bonita <- kable(tabla_bonita, 
                           format = "latex", 
                           digits = 4,
                           escape = FALSE,  # Permitir saltos de línea
                           caption = "Comparación de modelos predictivos")

# Guardar ambas versiones LaTeX
writeLines(tabla_latex_completa, "views/tabla_modelos_completa.tex")
writeLines(tabla_latex_bonita, "views/tabla_modelos_bonita.tex")

cat("✅ Tabla LaTeX completa exportada a: views/tabla_modelos_completa.tex\n")
cat("✅ Tabla LaTeX bonita exportada a: views/tabla_modelos_bonita.tex\n")

# Crear resumen ejecutivo
cat("\n=== RESUMEN EJECUTIVO ===\n")
cat("El mejor modelo según MSE es:", tabla_final$modelo[1], "\n")
if(tabla_final$Hiperparametro[1] != "-") {
  cat("Con hiperparámetro", tabla_final$Hiperparametro[1], "=", tabla_final$Valor_Hiperparametro[1], "\n")
}
cat("MSE =", tabla_final$MSE[1], ", R² =", tabla_final$R_squared[1], "\n")



