# Classificação de CBOs

Este projeto realiza classificação automática de ocupações da CBO (Classificação Brasileira de Ocupações) quanto a diversos fatores que podem influenciar na rotatividade de funcionários.

## Pré-requisitos

- Python 3.8 ou superior
- Arquivo `lista-cbo.csv` com dados de CBOs
- API key da OpenAI
- Bibliotecas Python: pandas, langchain, openai, tiktoken, tqdm, dotenv

## Instalação

1. Clone este repositório ou baixe os arquivos

2. Instale as dependências:
```bash
pip install pandas langchain-core langchain-openai openai tqdm python-dotenv tiktoken
```

3. Crie um arquivo `.env` na raiz do projeto e adicione sua chave API da OpenAI:
```
OPENAI_API_KEY=sua_chave_api_aqui
```

Exemplo de conteúdo do arquivo .env:
```
# Substitua com sua chave de API OpenAI
OPENAI_API_KEY=sk-your_openai_api_key_here

# Configurações adicionais (opcionais)
# MODEL=gpt-4o  # Modelo a ser usado, padrão é gpt-4o
```

## Estrutura do arquivo lista-cbo.csv

O arquivo `lista-cbo.csv` deve conter no mínimo as seguintes colunas:
- `codigo`: O código CBO
- `termo`: A descrição da ocupação
- `tipo` (opcional): Se presente, o script filtrará apenas entradas com tipo="Ocupação"

## Uso

Execute o script:
```bash
python classifica-cbo.py
```

O script irá:
1. Solicitar o número máximo de CBOs a processar (digite 0 para processar todos)
2. Carregar e preparar os dados da lista de CBOs
3. Enviar cada CBO para análise pela API da OpenAI
4. Processar as respostas e compilar os resultados
5. Salvar os resultados em formato CSV e Excel

## Arquivos de saída

- `cbos_classificados.csv` e `cbos_classificados.xlsx`: Contêm a classificação de cada CBO
- `token_usage_cbo.csv` e `token_usage_cbo.xlsx`: Contêm informações sobre uso de tokens e custos

## Formato dos resultados

Para cada CBO, o sistema gera um JSON com as seguintes propriedades:
- `análise`: Análise detalhada sobre as características da ocupação
- `trabalho_extenuante_estressante`: "sim", "não" ou "moderado"
- `baixa_escolaridade_exigida`: "sim", "não" ou "moderado"
- `sazonalidade`: "sim", "não" ou "moderado"
- `alta_exigencia_tecnica`: "sim", "não" ou "moderado"
- `salario_mais_elevado`: "sim", "não" ou "moderado"
- `interesse_em_reter_talentos`: "sim", "não" ou "moderado"
- `alta_rotatividade`: "sim", "não" ou "moderado"

Além disso, o sistema adiciona:
- `cbo`: O código CBO analisado
- `descricao`: A descrição da ocupação

## Considerações sobre custos

O script calcula e exibe os custos estimados de uso da API da OpenAI, baseados nos seguintes valores:
- Input: $2.5 por milhão de tokens
- Output: $10.0 por milhão de tokens

Estes valores são para o modelo gpt-4o e podem variar conforme a política de preços da OpenAI. 