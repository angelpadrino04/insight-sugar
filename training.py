
import json
from matplotlib import pyplot as plt
import numpy as np
from data_processing import load_and_preprocess_data
from mlp import MLP
import seaborn as sns


def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data()

    input_size = X_train.shape[1]
    model_configs = [
        {
            'hidden_layers': [16, 8],
            'hidden_activation': 'relu',
            'output_activation': 'sigmoid',
            'learning_rate': 0.01,
            'epochs': 500,
            'batch_size': 32,
            'early_stopping': 20,
            'random_seed': 42
        }
    ]

    for i, config in enumerate(model_configs):
        print(f"\nEntrenando modelo con configuración {i+1}:")
        print(config)

        model = MLP(
            input_size=input_size,
            hidden_layers=config['hidden_layers'],
            hidden_activation=config['hidden_activation'],
            output_activation=config['output_activation'],
            learning_rate=config['learning_rate'],
            random_seed=config['random_seed']
        )

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(config['epochs']):
            for j in range(0, X_train.shape[0], config['batch_size']):
                X_batch = X_train[j:j+config['batch_size']]
                y_batch = y_train[j:j+config['batch_size']]

                output = model.forward(X_batch)

                model.backward(X_batch, y_batch, output)

                model.update_params()

            train_output = model.forward(X_train)
            train_loss = model.compute_loss(y_train, train_output)
            train_losses.append(train_loss)

            val_output = model.forward(X_val)
            val_loss = model.compute_loss(y_val, val_output)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model.save_weights(f'results/best_model_config_{i+1}.json')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == config['early_stopping']:
                    print(f"Early stopping en época {epoch+1}")
                    break

            if (epoch + 1) % 50 == 0:
                print(
                    f"Época {epoch+1}/{config['epochs']} - Pérdida entrenamiento: {train_loss:.4f}, Pérdida validación: {val_loss:.4f}")

        model = MLP.load_weights(f'results/best_model_config_{i+1}.json')

        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test)

        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'config': {
                'input_size': input_size,
                **config
            }
        }

        with open(f'results/metrics_config_{i+1}.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        generate_plots(model, train_losses, val_losses,
                       metrics, i+1, X_test, y_test)

        generate_report(metrics, i+1)

        print("\nMétricas de evaluación:")
        print(
            f"Precisión - Train: {train_metrics['accuracy']:.4f}, Val: {val_metrics['accuracy']:.4f}, Test: {test_metrics['accuracy']:.4f}")
        print(
            f"Recall - Train: {train_metrics['recall']:.4f}, Val: {val_metrics['recall']:.4f}, Test: {test_metrics['recall']:.4f}")
        print(
            f"F1-score - Train: {train_metrics['f1_score']:.4f}, Val: {val_metrics['f1_score']:.4f}, Test: {test_metrics['f1_score']:.4f}")
        print(
            f"AUC - Train: {train_metrics['roc_auc']:.4f}, Val: {val_metrics['roc_auc']:.4f}, Test: {test_metrics['roc_auc']:.4f}")


def generate_plots(model, train_losses, val_losses, metrics, config_num, X_test, y_test):
    # Curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Entrenamiento')
    plt.plot(val_losses, label='Validación')
    plt.title(f'Curva de Aprendizaje - Configuración {config_num}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (Binary Cross-Entropy)')
    plt.legend()
    plt.savefig(f'plots/learning_curve_config_{config_num}.png')
    plt.close()

    cm = np.array(metrics['test']['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'Matriz de Confusión - Configuración {config_num}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(f'plots/confusion_matrix_config_{config_num}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics['test']['fpr'], metrics['test']['tpr'],
             label=f'ROC curve (AUC = {metrics["test"]["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - Configuración {config_num}')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/roc_curve_config_{config_num}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    y_prob = model.forward(X_test)
    plt.hist(y_prob[y_test == 0], bins=20, alpha=0.5, label='No Diabetes')
    plt.hist(y_prob[y_test == 1], bins=20, alpha=0.5, label='Diabetes')
    plt.title(f'Histograma de Predicciones - Configuración {config_num}')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig(f'plots/prediction_histogram_config_{config_num}.png')
    plt.close()


def generate_report(metrics, config_num):
    config = metrics['config']
    report = f"""# Análisis de Resultados - Clasificación de Diabetes (Configuración {config_num})

## Configuración del Modelo
- Arquitectura: {config.get('input_size', '?')}-{"-".join(map(str, config['hidden_layers']))}-{config.get('output_size', 1)}
- Funciones de activación: {metrics['config']['hidden_activation'].upper()} (ocultas), {metrics['config']['output_activation'].upper()} (salida)
- Tasa de aprendizaje: {metrics['config']['learning_rate']}
- Épocas: {metrics['config']['epochs']}
- Batch size: {metrics['config']['batch_size']}
- Early stopping: {metrics['config']['early_stopping']} épocas sin mejora

## Métricas de Rendimiento
| Métrica       | Entrenamiento | Validación | Prueba  |
|---------------|---------------|------------|---------|
| Precisión     | {metrics['train']['accuracy']:.4f} | {metrics['validation']['accuracy']:.4f} | {metrics['test']['accuracy']:.4f} |
| Recall        | {metrics['train']['recall']:.4f} | {metrics['validation']['recall']:.4f} | {metrics['test']['recall']:.4f} |
| F1-score      | {metrics['train']['f1_score']:.4f} | {metrics['validation']['f1_score']:.4f} | {metrics['test']['f1_score']:.4f} |
| AUC           | {metrics['train']['roc_auc']:.4f} | {metrics['validation']['roc_auc']:.4f} | {metrics['test']['roc_auc']:.4f} |

## Matriz de Confusión (Conjunto de Prueba)
Verdadero Negativo: {metrics['test']['confusion_matrix'][0][0]}
Falso Positivo: {metrics['test']['confusion_matrix'][0][1]}
Falso Negativo: {metrics['test']['confusion_matrix'][1][0]}
Verdadero Positivo: {metrics['test']['confusion_matrix'][1][1]}


## Conclusiones
El modelo logró un rendimiento {'aceptable' if metrics['test']['accuracy'] > 0.7 else 'modesto'} para la clasificación de diabetes, con una precisión de {metrics['test']['accuracy']:.2f} en el conjunto de prueba.

Las métricas de recall ({metrics['test']['recall']:.2f}) y F1-score ({metrics['test']['f1_score']:.2f}) sugieren que el modelo tiene {'un buen balance' if abs(metrics['test']['precision'] - metrics['test']['recall']) < 0.15 else 'cierto desbalance'} entre precisión y exhaustividad.

El área bajo la curva ROC (AUC = {metrics['test']['roc_auc']:.2f}) indica que el modelo tiene {'buena' if metrics['test']['roc_auc'] > 0.8 else 'moderada' if metrics['test']['roc_auc'] > 0.7 else 'limitada'} capacidad para distinguir entre las clases.

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
"""

    with open(f'results/analysis_report_config_{config_num}.md', 'w') as f:
        f.write(report)
