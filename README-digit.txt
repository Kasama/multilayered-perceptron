Autor: Roberto Alegro - 8936756

O Problema de aprendizado sobre o dataset mnist foi resolvido usando um modelo de multilayer perceptron, como visto em sala.

Requisitos do sistema:
	- python 3.x
	- numpy

Para executar o treinamento do modelo para esse exercicio, basta executar:
python digit_recognizer.py digit train 50

O programa executará o treinamento com 50 neuronios na camada escondida.
É necessário ter o CSV de treinamento localizado na pasta ./data/digit/train.csv

Para executar a predição dos resultados, basta executar:
python digit_recognizer.py digit predict 50

O programa executará a predição dos casos teste, imprimindo na saída padrão os resultados.
