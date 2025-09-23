## Taller 2 Ciencia de Datos y Econometría Aplicada - Equipo 6 - CheMarket Inc
## Gianluca Cicco - 202020881
## Parte B - Árboles causales y heterogeneidad de efectos

## 1. Arbol Causal para estimar efector heterogéneos del tratamiento sign_up
## 2. Identificar subgrupos en los que el efecto sea mas alto o mas bajo
## 3. Comparen estos hallazgos con los resultados promedio obtenidos en el taller 1: ¿el árbol confirma o matiza lo que se habia concluido antes?
## 4. Discutan la plausibilidad de los patrones encontrados y que implicaciónes tiene para la estrategia de la empresa

# --------------------------------------------------------------------------------
# 1. Arbol Causal para estimar efector heterogéneos del tratamiento sign_up
# --------------------------------------------------------------------------------

require(pacman)
p_load(htetree, rpart, dplyr, rpart.plot, caret, devtools, stargazer)

setwd("/Users/gianlucacicco/Desktop/2025-2/Ciencia de Datos/Taller 1")
db <- readRDS("/Users/gianlucacicco/Desktop/2025-2/Ciencia de Datos/Taller 1/Parte_B.Rds")

# La diferencia entre la parte A y B, es que en la A predigo Revenue, mientras que en la B estimo como sign_up afecta causalmente a Revenue, identificando
# subgrupos donde esta este efecto heterogéneo.

# Para tener arboles causales con propiedades estadisticas validas, tanto el split como la validación cruzada deben ser "honest", xval es la 
# validación cruzada, en este caso en 5 partes. Bucket hace referencia a "bucketing" que es agrupar las variables por grupos, por ejemplo, 25-28 en vez de
# ir individualmente por el 25, 26, 27 y 28. Treatment en este caso es signup, propensity es la probabilidad de estar en un grupo de tratamiento y minsize es
# el tamaño minimo de nodos terminales.

# Mi (y) seria revenue, mi treatment seria sign_up y las x las variables a las que tengo acceso

## Libreria "htetree"
setwd("/Users/gianlucacicco/Desktop/2025-2/Ciencia de Datos/Taller 2")

tree <- causalTree(Revenue ~ time_spent + past_sessions + device_type + os_type + is_returning_user,
                     data = db,
                     treatment = db$sign_up,
                     split.Rule = "CT", cv.option = "CT", 
                     split.Honest = TRUE, cv.Honest = TRUE,
                     split.Bucket = FALSE, xval = 5,
                     cp = 0, minsize = 20, propensity = 0.5)
opcp <- tree$cptable[,1][which.min(tree$cptable[,4])]
opfit <- prune(tree, opcp)
rpart.plot(opfit)

jpeg("arbol_causal.jpg", width = 1200, height = 800, res = 300, quality = 100)
rpart.plot(opfit)
dev.off()

## Libreria "casualTree" de Susan Athey
install_github("susanathey/causalTree")

tree2 <- causalTree(Revenue ~ time_spent + past_sessions + device_type + os_type + is_returning_user,
                   data = db,
                   treatment = db$sign_up,
                   split.Rule = "CT", cv.option = "CT", 
                   split.Honest = TRUE, cv.Honest = TRUE,
                   split.Bucket = FALSE, xval = 5,
                   cp = 0, minsize = 20, propensity = 0.5)

opcp2 <- tree2$cptable[,1][which.min(tree$cptable[,4])]
opfit2 <- prune(tree, opcp)
rpart.plot(opfit2)

jpeg("arbol_causal_2.jpg", width = 1200, height = 800, res = 300, quality = 100)
rpart.plot(opfit, main = "Efectos Heterogéneos de sign_up sobre Revenue")
dev.off()

# --------------------------------------------------------------------------------
# 2. Identificar subgrupos en los que el efecto sea mas alto o mas bajo
# --------------------------------------------------------------------------------

# Capturar las reglas reales del árbol
reglas_output <- capture.output(rpart.rules(opfit, style = "wide", cover = TRUE))

# Limpiar el output (quitar líneas vacías y formatear)
reglas_limpias <- reglas_output[reglas_output != ""]
reglas_limpias <- reglas_limpias[!grepl("^\\s*$", reglas_limpias)]

# Obtener predicciones únicas ordenadas
predicciones <- predict(opfit)
efectos_unicos <- sort(unique(predicciones), decreasing = TRUE)

# Crear tabla con reglas reales
tabla_con_reglas <- data.frame()

for(i in 1:length(efectos_unicos)) {
  efecto <- efectos_unicos[i]
  n_usuarios <- sum(predicciones == efecto)
  
  # Intentar obtener la regla específica
  # (Las reglas de rpart.rules están ordenadas pero puede variar)
  if(i <= length(reglas_limpias)) {
    regla_texto <- reglas_limpias[i]
    # Limpiar formato si es necesario
    regla_texto <- gsub("^\\s+", "", regla_texto)  # Quitar espacios iniciales
    regla_texto <- gsub("\\s+", " ", regla_texto)   # Normalizar espacios
  } else {
    regla_texto <- paste0("Subgrupo ", i, " (efecto = ", round(efecto, 3), ")")
  }
  
  categoria <- if(i <= 8) "Top 8 (Alto)" else if(i > length(efectos_unicos) - 8) "Bottom 8 (Bajo)" else "Intermedio"
  
  fila <- data.frame(
    Ranking = i,
    Regla = regla_texto,
    Efecto_Causal = round(efecto, 3),
    N_Usuarios = n_usuarios,
    Categoria = categoria,
    stringsAsFactors = FALSE
  )
  
  tabla_con_reglas <- rbind(tabla_con_reglas, fila)
}

# Filtrar extremos
tabla_extremos <- tabla_con_reglas[tabla_con_reglas$Categoria != "Intermedio", ]

# Mostrar las reglas en consola primero para verificar
cat("=== VERIFICACIÓN: REGLAS DEL ÁRBOL ===\n")
for(i in 1:min(5, length(reglas_limpias))) {
  cat("Regla", i, ":", reglas_limpias[i], "\n")
}

# Exportar tabla
stargazer(tabla_extremos, 
          type = "html",
          summary = FALSE,
          title = "Árbol Causal: Reglas Específicas para Top 8 y Bottom 8",
          digits = 3,
          rownames = FALSE,
          column.labels = c("Ranking", "Regla del Árbol", "Efecto Causal", "N° Usuarios", "Categoría"),
          out = "tabla_reglas_completas.html")
