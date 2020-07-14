# Trabalho Final machine-leaning-2020.1

* Dataset: https://www.kaggle.com/ferraz/acordaos-tcu
* Objetivo:
    * O objetivo deste experimento é avaliar o desempenho de diferentes classificadores e representações textuais na resposta a seguinte pergunta:
        * Quem é foi o relator responsável pelo acórdão?
* Etapas:
    * Análise Exploratória dos Dados
    * Representações do texto: TF-IDF, word2Vec pré-treinado e word2vec treinado na base
    * Classificadores: KNN, SVM e MLP(sklearn)
    * Avaliação de Resultados com Testes de Hipótese

## Como usar?

* Requerimento: ter o Docker instalado

* Construir a imagem docker: 
` docker build -t ml .`

* Criar container docker: 
` docker run -itd --rm -v ${PWD}:/app -w /app -p 8889:8888 --name ml ml bash `

* Acessar o container:  
`docker exec -it ml bash`

* Para executar o experimento, após acessar o container basta executar o seguinte comando:
`python src/experiment.py`

* Iniciar jupyternotebook: 
`jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='ml' &`

* Para acessar o jupyter notebook, acessar a url (http://localhost:8889/lab) no browser e usar o token 'ml'
