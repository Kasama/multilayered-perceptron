Autor: Roberto Alegro - 8936756

O Problema de aprendizado sobre o dataset do titanic foi resolvido usando um modelo de multilayer perceptron, como visto em sala.

Requisitos do sistema:
	- python 3.x
	- numpy
	- pandas

Para executar o treinamento do modelo para esse exercicio, basta executar:
python titanic.py titanic train 5

O programa executará o treinamento com 5 neuronios na camada escondida.
É necessário ter o CSV de treinamento localizado na pasta ./data/titanic/train.csv

Para executar a predição dos resultados, basta executar:
python titanic.py titanic predict 5

O programa executará a predição dos casos teste, imprimindo na saída padrão os resultados.
