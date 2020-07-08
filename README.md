# Trabalho Final machine-leaning-2020.1

* Dataset: https://www.kaggle.com/ferraz/acordaos-tcu
* Objetivo:
    * Quem é foi o relator responsável pelo acórdão?
    * Qual o tipo de processo do acórdão?
* Como:
    * Análise exploratória dos dados
    * Representações do texto: TF-IDF, word2Vec, https://github.com/IBM/WordMoversEmbeddings, https://github.com/UKPLab/sentence-transformers, https://github.com/XiaoqiJiao/COLING2018 - outros(https://github.com/Separius/awesome-sentence-embedding)
    * Classificador: SVM, MLP(sklearn), Custom NN (tf.keras in TensorFlow 2.0)[https://www.tensorflow.org/guide/keras]
    * Testes de Hipótese

## Para executar os notebooks
* Buildar a imagem docker: docker build -t ml .
* Criar container docker: docker run -itd --rm -v ${PWD}:/app -w /app -p 8889:8888 --name ml ml bash
* Acessar o container:  docker exec -it ml bash
* Iniciar jupyternotebook: jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='ml' &
* Acessar no browser: http://localhost:8888/lab

## Requerimentos
- Docker