# Análisis de Resultados - Clasificación de Diabetes (Configuración 1)

## Configuración del Modelo
- Arquitectura: 8-16-8-1
- Funciones de activación: RELU (ocultas), SIGMOID (salida)
- Tasa de aprendizaje: 0.01
- Épocas: 500
- Batch size: 32
- Early stopping: 20 épocas sin mejora

## Métricas de Rendimiento
| Métrica       | Entrenamiento | Validación | Prueba  |
|---------------|---------------|------------|---------|
| Precisión     | 0.8130 | 0.7922 | 0.7468 |
| Recall        | 0.6747 | 0.6600 | 0.5769 |
| F1-score      | 0.7226 | 0.6735 | 0.6061 |
| AUC           | 0.8690 | 0.8344 | 0.8303 |

## Matriz de Confusión (Conjunto de Prueba)
Verdadero Negativo: 85
Falso Positivo: 17
Falso Negativo: 22
Verdadero Positivo: 30


## Conclusiones
El modelo logró un rendimiento aceptable para la clasificación de diabetes, con una precisión de 0.75 en el conjunto de prueba.

Las métricas de recall (0.58) y F1-score (0.61) sugieren que el modelo tiene un buen balance entre precisión y exhaustividad.

El área bajo la curva ROC (AUC = 0.83) indica que el modelo tiene buena capacidad para distinguir entre las clases.

## Limitaciones y Posibles Mejoras
1. **Limitaciones**:
   - El tamaño del dataset es relativamente pequeño (768 muestras)
   - Posible desbalance de clases (aproximadamente 35% diabéticos, 65% no diabéticos)
   - Las características pueden no capturar completamente los factores de riesgo de diabetes

2. **Posibles Mejoras**:
   - Aumentar el tamaño del dataset mediante técnicas de aumento de datos
   - Implementar balanceo de clases (oversampling, undersampling o class weighting)
   - Probar arquitecturas más complejas o diferentes funciones de activación
   - Optimizar hiperparámetros mediante búsqueda en grid o random search
   - Añadir regularización (L1/L2) o dropout para reducir overfitting
