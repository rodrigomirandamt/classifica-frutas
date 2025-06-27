# Como Executar Prompts em Escala com LangChain Runnables e OpenAI

Refira para o folder CBO-RUNNABLES para um codigo em produto
e no folder como-executar-prompts-em-escala/pratique para um codigo simples de pratica

Este guia apresenta como utilizar **LangChain Runnables** com a **OpenAI** para executar prompts em escala de forma eficiente, com base no projeto de classificação de códigos CBO.

* Para um exemplo de aplicação em produção, consulte o diretório `CBO-RUNNABLES`.
* Para um exemplo simples e didático, acesse o diretório `como-executar-prompts-em-escala/pratique`.

## Por que usar Runnables para Processamento em Escala?

Quando você tem **dezenas de milhares de prompts** para processar ou precisa criar **milhares de prompts customizados**, o processamento sequencial se torna inviável. Imagine cenários como:

- **Classificar 50.000 CBOs** com análises personalizadas
- **Processar milhares de documentos** para extração de dados
- **Gerar conteúdo personalizado** para dezenas de milhares de usuários
- **Analisar grandes datasets** com prompts específicos para cada registro

**Sem Runnables**: Processar 10.000 prompts sequencialmente levaria ~10 horas (assumindo 3.6s por prompt)
**Com Runnables**: O mesmo processamento pode ser reduzido para ~1-2 horas usando paralelização

### Benefícios dos Runnables:

- ⚡ **Performance**: Processamento paralelo automático
- 💰 **Economia**: Melhor controle de custos e tokens
- 🔄 **Robustez**: Tratamento de erros e retry automático
- 📊 **Monitoramento**: Tracking detalhado de progresso
- 🔧 **Flexibilidade**: Componentes reutilizáveis e composáveis

## Visão Geral dos Passos

Este guia está organizado em etapas progressivas para você dominar o processamento em escala:

### 🎯 **Fundamentos (Passos 1-3)**

- **Configuração**: Setup do ambiente e dependências
- **Básicos**: Criação de chains simples com Runnables
- **Composição**: Combinação de componentes usando operador pipe (`|`)

### ⚡ **Paralelização (Passos 4-5)**

- **RunnableMap**: Processamento paralelo de múltiplos inputs
- **RunnableParallel**: Execução simultânea de diferentes análises

### 💡 **Otimização (Passos 6-8)**

- **Controle de Tokens**: Monitoramento e cálculo de custos
- **Streaming**: Processamento assíncrono e em tempo real
- **Configuração Avançada**: Callbacks, tags e metadados

### 🏗️ **Produção (Passos 9-11)**

- **Exemplo Completo**: Classe para processamento industrial
- **Melhores Práticas**: Rate limiting, cache e tratamento de erros
- **Monitoramento**: Debugging e logging em produção

## O que são Runnables?

**Runnables** são a interface padrão do LangChain para criar componentes composáveis e interconectáveis. Eles implementam métodos como `invoke`, `batch`, `stream` e fornecem a base para construir pipelines de processamento de linguagem natural.

### Principais características dos Runnables:

- **Composição**: Podem ser combinados usando o operador `|` (pipe)
- **Configuração**: Suportam configuração via `RunnableConfig`
- **Paralelização**: Executam operações de forma eficiente
- **Streaming**: Suportam saída em tempo real
- **Callbacks**: Propagam callbacks automaticamente

## Pré-requisitos

```bash
pip install langchain-core langchain-openai openai tqdm python-dotenv tiktoken
```

## Configuração do Ambiente

### 1. Arquivo .env

Crie um arquivo `.env` na raiz do projeto:

```env
# Chave de API OpenAI (obrigatória)
OPENAI_API_KEY=sk-your_openai_api_key_here

# Modelo a ser usado (opcional, padrão: gpt-4o)
MODEL=gpt-4o

# Configurações de temperatura (opcional)
TEMPERATURE=0
```

### 2. Importações Básicas

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import tiktoken

# Carregar variáveis de ambiente
load_dotenv()
```

## Estrutura Básica de um Runnable com OpenAI

### 1. Inicializando o Modelo OpenAI

```python
# Obter configurações do ambiente
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL", "gpt-4o")

if not api_key:
    raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")

# Inicializar o modelo OpenAI como Runnable
model = ChatOpenAI(
    model=model_name, 
    temperature=0, 
    api_key=api_key
)

print(f"Usando modelo: {model_name}")
```

**💡 Sumário:** Configuramos o modelo ChatOpenAI como um Runnable, obtendo as credenciais do arquivo .env e definindo parâmetros como temperatura. Este modelo será a base de todas as chains que criaremos.

### 2. Criando Templates de Prompt

```python
# Template básico para análise de CBO
def create_cbo_prompt_template():
    template = """
    Analise o CBO {cbo_code} ({description}) e responda em formato JSON:
  
    {{
      "análise": "Análise detalhada da ocupação",
      "trabalho_extenuante_estressante": "sim/não/moderado",
      "baixa_escolaridade_exigida": "sim/não/moderado",
      "sazonalidade": "sim/não/moderado",
      "alta_exigencia_tecnica": "sim/não/moderado",
      "salario_mais_elevado": "sim/não/moderado",
      "interesse_em_reter_talentos": "sim/não/moderado",
      "alta_rotatividade": "sim/não/moderado"
    }}
    """
  
    return ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em análise de mercado de trabalho brasileiro."),
        ("human", template)
    ])
```

**💡 Sumário:** Criamos um template reutilizável que define a estrutura do prompt e o formato de resposta esperado. O template usa variáveis (`{cbo_code}`, `{description}`) que serão preenchidas dinamicamente para cada CBO.

### 3. Compondo uma Chain Básica

```python
# Criar template de prompt
prompt_template = create_cbo_prompt_template()

# Criar parser de saída
output_parser = StrOutputParser()

# Compor a chain usando o operador pipe (|)
chain = prompt_template | model | output_parser

# Invocar a chain
result = chain.invoke({
    "cbo_code": "2142-05", 
    "description": "Engenheiro de software"
})

print(result)
```

**💡 Sumário:** Combinamos três componentes (prompt template + modelo + parser) usando o operador pipe (`|`) para criar uma chain completa. Esta chain pode ser invocada com dados e retornará a resposta processada e formatada.

## Processamento em Paralelo com RunnableMap

### 1. Processamento Múltiplo

```python
# Função para criar entrada individual
def create_input(cbo_code, description):
    return {
        "cbo_code": cbo_code,
        "description": description
    }

# Lista de CBOs para processar
cbos_data = [
    ("2142-05", "Engenheiro de software"),
    ("3171-05", "Técnico em informática"),
    ("5111-05", "Vendedor de comércio varejista")
]

# Preparar inputs
inputs = [create_input(code, desc) for code, desc in cbos_data]

# Usar map() para processamento paralelo
map_chain = chain.map()

# Processar todos os inputs em paralelo
results = map_chain.invoke(inputs)

for i, result in enumerate(results):
    print(f"CBO {cbos_data[i][0]}: {result[:100]}...")
```

**💡 Sumário:** Usando `chain.map()`, transformamos nossa chain simples em um processador paralelo que pode lidar com múltiplos inputs simultaneamente, drasticamente reduzindo o tempo total de processamento.

### 2. Processamento com RunnableParallel

```python
from langchain_core.runnables import RunnableParallel

# Criar chains paralelas para diferentes análises
analysis_chain = RunnableParallel({
    "basic_analysis": prompt_template | model | output_parser,
    "token_count": RunnableLambda(lambda x: count_tokens(str(x))),
    "metadata": RunnableLambda(lambda x: {
        "cbo": x["cbo_code"],
        "timestamp": "2024-01-01"
    })
})

# Executar análises paralelas
parallel_result = analysis_chain.invoke({
    "cbo_code": "2142-05", 
    "description": "Engenheiro de software"
})

print("Análise:", parallel_result["basic_analysis"])
print("Tokens:", parallel_result["token_count"])
print("Metadata:", parallel_result["metadata"])
```

**💡 Sumário:** RunnableParallel permite executar diferentes análises simultaneamente no mesmo input, retornando um dicionário com todos os resultados. Ideal para quando você precisa de múltiplas perspectivas dos mesmos dados.

## Controle de Tokens e Custos

### 1. Função de Contagem de Tokens

```python
def count_tokens(text, model="gpt-4o"):
    """Conta tokens usando tiktoken"""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))

# Criar Runnable para controle de tokens
def create_token_aware_chain():
    def process_with_tokens(inputs):
        # Contar tokens de entrada
        input_text = str(inputs)
        input_tokens = count_tokens(input_text)
    
        # Processar com a chain
        result = chain.invoke(inputs)
    
        # Contar tokens de saída
        output_tokens = count_tokens(result)
    
        return {
            "result": result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
  
    return RunnableLambda(process_with_tokens)

# Usar chain com controle de tokens
token_chain = create_token_aware_chain()
token_result = token_chain.invoke({
    "cbo_code": "2142-05", 
    "description": "Engenheiro de software"
})

print(f"Tokens de entrada: {token_result['input_tokens']}")
print(f"Tokens de saída: {token_result['output_tokens']}")
print(f"Total: {token_result['total_tokens']}")
```

**💡 Sumário:** Criamos uma função wrapper que conta tokens de entrada e saída usando tiktoken, permitindo monitoramento preciso do uso de tokens para controle de custos e performance.

### 2. Cálculo de Custos

```python
def calculate_costs(input_tokens, output_tokens, model_name="gpt-4o"):
    """Calcula custos baseado no modelo"""
    pricing = {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-4": {"input": 30.0, "output": 60.0}
    }
  
    rates = pricing.get(model_name, {"input": 2.5, "output": 10.0})
  
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
  
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

# Integrar cálculo de custos
def create_cost_aware_chain():
    def process_with_costs(inputs):
        token_result = token_chain.invoke(inputs)
        costs = calculate_costs(
            token_result['input_tokens'],
            token_result['output_tokens'],
            model_name
        )
    
        return {
            **token_result,
            **costs
        }
  
    return RunnableLambda(process_with_costs)
```

**💡 Sumário:** Implementamos um sistema de cálculo de custos baseado em diferentes modelos OpenAI, convertendo o número de tokens em valores monetários para melhor controle orçamentário.

## Streaming e Processamento Assíncrono

### 1. Streaming de Resultados

```python
# Streaming síncrono
for chunk in chain.stream({
    "cbo_code": "2142-05", 
    "description": "Engenheiro de software"
}):
    print(chunk, end="", flush=True)
```

**💡 Sumário:** O streaming permite ver resultados em tempo real conforme são gerados, útil para respostas longas ou quando você quer feedback imediato do progresso.

### 2. Processamento Assíncrono

```python
import asyncio

async def async_processing():
    # Processamento assíncrono individual
    result = await chain.ainvoke({
        "cbo_code": "2142-05", 
        "description": "Engenheiro de software"
    })
    print(result)
  
    # Processamento em lote assíncrono
    results = await chain.abatch(inputs)
    return results

# Executar processamento assíncrono
async def main():
    results = await async_processing()
    print(f"Processados {len(results)} CBOs")

# asyncio.run(main())
```

**💡 Sumário:** O processamento assíncrono com `ainvoke` e `abatch` permite máxima eficiência ao processar múltiplos requests simultâneos, especialmente importante para grandes volumes de dados.

## Configuração Avançada com RunnableConfig

### 1. Callbacks e Monitoramento

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
  
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            tokens = usage.get('total_tokens', 0)
            self.total_tokens += tokens
            print(f"Tokens usados: {tokens}")

# Usar callback
callback = TokenUsageCallback()

result = chain.invoke(
    {"cbo_code": "2142-05", "description": "Engenheiro de software"},
    config={"callbacks": [callback]}
)

print(f"Total de tokens: {callback.total_tokens}")
```

**💡 Sumário:** Callbacks permitem interceptar e monitorar eventos durante a execução das chains, essencial para logging, debugging e coleta de métricas em ambiente de produção.

### 2. Tags e Metadados

```python
# Configurar chain com tags e metadados
result = chain.invoke(
    {"cbo_code": "2142-05", "description": "Engenheiro de software"},
    config={
        "tags": ["cbo-analysis", "production"],
        "metadata": {"batch_id": "batch_001", "priority": "high"}
    }
)
```

**💡 Sumário:** Tags e metadados permitem categorizar e rastrear execuções específicas, facilitando análise posterior e organização de logs em sistemas de monitoramento.

## Exemplo Completo: Processamento em Escala

```python
import pandas as pd
from tqdm import tqdm

class CBOProcessor:
    def __init__(self, model_name="gpt-4o", batch_size=10):
        self.model_name = model_name
        self.batch_size = batch_size
        self.setup_chain()
    
    def setup_chain(self):
        """Configura a chain de processamento"""
        api_key = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(model=self.model_name, temperature=0, api_key=api_key)
    
        prompt_template = create_cbo_prompt_template()
        output_parser = StrOutputParser()
    
        self.chain = prompt_template | model | output_parser
        self.map_chain = self.chain.map()
  
    def process_dataframe(self, df):
        """Processa DataFrame com dados de CBO"""
        # Preparar inputs
        inputs = []
        for _, row in df.iterrows():
            inputs.append({
                "cbo_code": row['codigo'],
                "description": row['termo']
            })
    
        # Processar em lotes
        results = []
        total_batches = len(inputs) // self.batch_size + (1 if len(inputs) % self.batch_size else 0)
    
        for i in tqdm(range(0, len(inputs), self.batch_size), desc="Processando lotes"):
            batch = inputs[i:i + self.batch_size]
            batch_results = self.map_chain.invoke(batch)
            results.extend(batch_results)
    
        return results
  
    def save_results(self, results, df, filename="resultados.csv"):
        """Salva resultados em arquivo"""
        results_df = pd.DataFrame(results)
        results_df['cbo'] = df['codigo'].tolist()[:len(results)]
        results_df['descricao'] = df['termo'].tolist()[:len(results)]
    
        results_df.to_csv(filename, index=False)
        print(f"Resultados salvos em {filename}")

# Uso da classe
if __name__ == "__main__":
    # Carregar dados
    df = pd.read_csv('lista-cbo.csv')
  
    # Inicializar processador
    processor = CBOProcessor(model_name="gpt-4o", batch_size=5)
  
    # Processar dados
    results = processor.process_dataframe(df.head(20))  # Teste com 20 registros
  
    # Salvar resultados
    processor.save_results(results, df.head(20))
```

**💡 Sumário:** Esta classe `CBOProcessor` encapsula todos os conceitos anteriores em uma solução pronta para produção, incluindo processamento em lotes, monitoramento de progresso e controle de recursos.

## Melhores Práticas

### 1. Gestão de Rate Limits

```python
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    def decorator(func):
        last_called = [0.0]
    
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 60.0 / calls_per_minute - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Aplicar rate limiting
@rate_limit(calls_per_minute=60)
def limited_invoke(chain, input_data):
    return chain.invoke(input_data)
```

**💡 Sumário:** Rate limiting é essencial para evitar exceder os limites da API OpenAI. Este decorator implementa um controle automático de frequência de chamadas.

**💡 Sumário:** Rate limiting é essencial para evitar exceder os limites da API OpenAI. Este decorator implementa um controle automático de frequência de chamadas.

### 2. Tratamento de Erros

```python
def create_resilient_chain():
    def safe_process(inputs):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"Erro ao processar {inputs.get('cbo_code', 'unknown')}: {e}")
            return {
                "error": str(e),
                "cbo_code": inputs.get('cbo_code'),
                "success": False
            }
  
    return RunnableLambda(safe_process)

resilient_chain = create_resilient_chain()
```

**💡 Sumário:** Tratamento robusto de erros evita que falhas pontuais interrompam o processamento completo, retornando informações estruturadas sobre problemas encontrados.

### 3. Cache de Resultados

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_invoke(cbo_code, description):
    return chain.invoke({
        "cbo_code": cbo_code,
        "description": description
    })
```

**💡 Sumário:** Cache LRU evita reprocessamento desnecessário de inputs idênticos, economizando tokens e tempo, especialmente útil quando há dados duplicados no dataset.

## Monitoramento e Debugging

### 1. Inspeção da Chain

```python
# Visualizar estrutura da chain
print(chain.get_graph().print_ascii())

# Obter prompts utilizados
prompts = chain.get_prompts()
for prompt in prompts:
    print(prompt)
```

**💡 Sumário:** Ferramentas de inspeção ajudam a entender a estrutura interna das chains e verificar se os prompts estão sendo construídos corretamente antes da execução.

### 2. Logging Detalhado

```python
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chain com logging
def logged_chain_invoke(inputs):
    logger.info(f"Processando CBO: {inputs.get('cbo_code')}")
    start_time = time.time()
  
    result = chain.invoke(inputs)
  
    duration = time.time() - start_time
    logger.info(f"Concluído em {duration:.2f}s")
  
    return result

logged_chain = RunnableLambda(logged_chain_invoke)
```

**💡 Sumário:** Logging estruturado fornece visibilidade completa sobre o processo de execução, incluindo tempos de resposta e identificação de gargalos de performance.

## Considerações de Performance

1. **Paralelização**: Use `map()` para processamento paralelo
2. **Batch Size**: Encontre o equilíbrio entre throughput e rate limits
3. **Caching**: Implemente cache para evitar reprocessamento
4. **Monitoring**: Monitore tokens e custos constantemente
5. **Error Handling**: Implemente retry logic e tratamento de falhas

## 🍎 Exemplo Prático

Para ver todos esses conceitos em ação, execute o arquivo `aprenda.py` que demonstra o processamento de 50 frutas:

```bash
python aprenda.py
```

Este exemplo mostra:

- Comparação entre processamento sequencial vs paralelo
- Tracking de tokens e custos em tempo real
- Processamento em lotes com barra de progresso
- Salvamento de resultados estruturados
- Estatísticas detalhadas de performance

📖 **Veja o arquivo `README-exemplo-frutas.md` para instruções detalhadas**

## Conclusão

Os LangChain Runnables oferecem uma interface poderosa e flexível para construir pipelines de processamento com OpenAI. A combinação de composabilidade, paralelização e configurabilidade torna possível processar prompts em escala de forma eficiente e controlada.

Para projetos de classificação como o de CBOs, essa abordagem permite:

- Processamento eficiente de grandes volumes de dados
- Controle preciso de custos e tokens
- Monitoramento detalhado do pipeline
- Facilidade de manutenção e extensão do código
