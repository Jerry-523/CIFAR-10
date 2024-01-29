# CIFAR-10
# Classification Models

Este repositório contém implementações e exemplos de quatro modelos diferentes para a classificação do conjunto de dados CIFAR-10: KNN, SVM, Softmax Regression e uma Rede Neural Convolucional (CNN) de duas camadas.

## KNN (K-Nearest Neighbors)

### Funcionamento
O KNN é um algoritmo de aprendizado supervisionado usado para classificação. A ideia principal é classificar um ponto de dados baseado na classe predominante entre seus k vizinhos mais próximos. A distância entre pontos é uma consideração crucial, e o hiperparâmetro k define o número de vizinhos considerados.

### Execução
- `knn_model = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='manhattan')`
- `knn_model.fit(train_images, train_labels)`
- `predicted_labels = knn_model.predict(test_images)`

## SVM (Support Vector Machine)

### Funcionamento
SVM é um algoritmo de aprendizado supervisionado usado para classificação. Ele encontra um hiperplano de decisão ótimo que separa as classes no espaço de características. A otimização visa maximizar a margem entre as classes.

### Execução
- `svm_model = SVC(kernel='linear', C=1.0)`
- `svm_model.fit(train_images, train_labels)`
- `predicted_labels = svm_model.predict(test_images)`

## Softmax Regression

### Funcionamento
Softmax Regression é uma generalização da regressão logística para problemas de múltiplas classes. Ele modela a probabilidade de uma determinada classe para cada ponto de dados e escolhe a classe com a maior probabilidade.

### Execução
- `softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)`
- `softmax_model.fit(train_images, train_labels)`
- `predicted_labels = softmax_model.predict(test_images)`

## CNN (Rede Neural Convolucional de 2 Camadas)

### Funcionamento
A CNN é uma arquitetura de rede neural profunda projetada para tarefas de visão computacional. Esta implementação usa duas camadas: uma camada convolucional seguida por uma camada totalmente conectada. A convolução permite a extração de características locais.

### Execução
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

