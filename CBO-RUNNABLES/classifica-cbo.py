import pandas as pd
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm
import unicodedata
import tiktoken
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Função para normalizar textos com acentos e caracteres especiais
def normalize_text(text):
    if isinstance(text, str):
        # Remover caracteres não imprimíveis
        text = ''.join(c for c in text if c.isprintable())
        # Normalizar acentos e caracteres especiais
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

# Função para contar tokens usando tiktoken
def count_tokens(text, model="gpt-4o"):
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except:
        # Fallback para cl100k_base se o modelo específico não for encontrado
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))

# Obter API key e modelo do arquivo .env
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL", "gpt-4o")  # Valor padrão é gpt-4o se não estiver definido

if not api_key:
    raise ValueError("API key da OpenAI não encontrada no arquivo .env. Verifique se o arquivo contém OPENAI_API_KEY=sua_chave_api")

print(f"Usando modelo: {model_name}")

# Load the CBO CSV file
print("Carregando arquivo lista-cbo.csv...")
cbo_data = pd.read_csv('lista-cbo.csv')

# Limpar e filtrar dados
print("Limpando e preparando os dados...")

# Verificar as colunas disponíveis
print(f"Colunas disponíveis no arquivo: {cbo_data.columns.tolist()}")

# Selecionar apenas as colunas necessárias
if 'tipo' in cbo_data.columns:
    cbo_data = cbo_data[['codigo', 'termo', 'tipo']]
else:
    cbo_data = cbo_data[['codigo', 'termo']]

# Remover linhas com valores nulos
cbo_data = cbo_data.dropna(subset=['codigo', 'termo'])

# Filtrar apenas entradas do tipo 'Ocupação' se a coluna 'tipo' estiver disponível
if 'tipo' in cbo_data.columns:
    cbo_data = cbo_data[cbo_data['tipo'] == 'Ocupação']

# Remover duplicatas no código CBO
cbo_data = cbo_data.drop_duplicates(subset=['codigo'])

# Debug: Verificar o formato dos dados
print(f"Exemplo de dados após limpeza:")
print(cbo_data.head())

# Exibir informações sobre os dados
print(f"Total de CBOs únicos para processar: {len(cbo_data)}")

# Inicializar o modelo OpenAI com LangChain
model = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)

# Função para criar prompt para classificação de CBO
def create_cbo_prompt(cbo_code, description):
    prompt = f"""
    Considere as seguintes características de ocupações:
    - Trabalho extenuante / estressante
    - Baixa escolaridade exigida
    - Sazonalidade
    - Alta exigência técnica
    - Salário mais elevado
    - Interesse da empresa em reter talentos

    Baseado nas informações conhecidas sobre o CBO {cbo_code} ({description}). Raciocine sobre o que você acha sobre a rotatividade do CBO {cbo_code} e
    em seguida responda o questionário em formato JSON, preenchendo "sim", "não" ou "moderado" para cada uma das características abaixo:

    Responda apenas com o JSON, sem texto adicional. Exceto na pergunta de analise, responder como sim, não ou moderado.
    {{
      "análise": "Análise detalhada sobre as características da ocupação relacionadas a rotatividade",
      "trabalho_extenuante_estressante": "sim/não/moderado",
      "baixa_escolaridade_exigida": "sim/não/moderado",
      "sazonalidade": "sim/não/moderado",
      "alta_exigencia_tecnica": "sim/não/moderado",
      "salario_mais_elevado": "sim/não/moderado",
      "interesse_em_reter_talentos": "sim/não/moderado",
      "alta_rotatividade": "sim/não/moderado"
    }}
    """
    return prompt

# Function to get user input for max CBOs to process
def get_max_cbos():
    while True:
        try:
            max_cbos = int(input("Digite o número máximo de CBOs a processar (0 para todos): "))
            if max_cbos < 0:
                print("Por favor, digite um número não negativo.")
            else:
                return max_cbos
        except ValueError:
            print("Por favor, digite um número inteiro válido.")

# Get max_cbos from user
max_cbos = get_max_cbos()

# Limit the number of CBOs to process if specified
cbos_to_process = cbo_data.head(max_cbos) if max_cbos > 0 else cbo_data

# Prepare prompts
print("Preparando prompts para processamento...")
prompts = []
total_input_tokens = 0

for _, row in cbos_to_process.iterrows():
    prompt = create_cbo_prompt(row['codigo'], row['termo'])
    prompts.append(prompt)
    total_input_tokens += count_tokens(prompt, model_name)

# Adicionar tokens do sistema
system_message = "Você é um assistente especializado em análise de mercado de trabalho e ocupações profissionais do Brasil."
system_tokens = count_tokens(system_message, model_name)
total_input_tokens += system_tokens * len(prompts)

print(f"Total de tokens de entrada (input): {total_input_tokens}")

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{prompt}")
])

# Create the chain
chain = prompt_template | model

# Prepare the RunnableMap for parallel processing
print("Configurando processamento paralelo...")
map_chain = chain.map()

# Process CBOs with progress bar
print("Processando CBOs em paralelo...")
inputs = [{"prompt": prompt} for prompt in prompts]

if len(inputs) > 0:
    raw_responses = map_chain.invoke(inputs)

    # Calculate output tokens
    total_output_tokens = 0
    for response in raw_responses:
        total_output_tokens += count_tokens(response.content, model_name)

    print(f"Total de tokens de saída (output): {total_output_tokens}")

    # Get model pricing (adjust as needed based on OpenAI's current pricing)
    model_pricing = {
        "gpt-4o": {"input": 2.5, "output": 10.0},  # $2.5 per million input tokens, $10 per million output tokens
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},  # $0.5 per million input tokens, $1.5 per million output tokens
        "gpt-4": {"input": 30.0, "output": 60.0}  # $30 per million input tokens, $60 per million output tokens
    }
    
    # Default pricing if model not in the dictionary
    default_pricing = {"input": 2.5, "output": 10.0}
    
    # Get pricing for the current model
    pricing = model_pricing.get(model_name, default_pricing)
    input_cost_per_million = pricing["input"]
    output_cost_per_million = pricing["output"]

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    print(f"Custo de tokens de entrada (${input_cost_per_million}/M): ${input_cost:.4f}")
    print(f"Custo de tokens de saída (${output_cost_per_million}/M): ${output_cost:.4f}")
    print(f"Custo total estimado: ${total_cost:.4f}")

    # Extract content from AIMessage and convert to classifications
    def parse_response(response):
        try:
            content = response.content
            
            # Check if response is in a markdown code block
            if "```json" in content:
                # Extract JSON from the markdown code block
                start_idx = content.find("```json") + 7
                end_idx = content.rfind("```")
                if end_idx > start_idx:
                    content = content[start_idx:end_idx].strip()
            elif "```" in content:
                # Extract JSON from generic markdown code block
                start_idx = content.find("```") + 3
                end_idx = content.rfind("```")
                if end_idx > start_idx:
                    content = content[start_idx:end_idx].strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
            print(f"Conteúdo da resposta: {response.content}")
            return None

    # Parse responses and add CBO info
    print("Processando respostas...")
    results = []
    for i, response in enumerate(raw_responses):
        if i < len(cbos_to_process):
            parsed = parse_response(response)
            if parsed:
                parsed["cbo"] = cbos_to_process.iloc[i]['codigo']
                parsed["descricao"] = cbos_to_process.iloc[i]['termo']
                results.append(parsed)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Adicionar informações de tokens e custos aos resultados
    token_info = {
        'modelo': model_name,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost_usd': input_cost,
        'output_cost_usd': output_cost,
        'total_cost_usd': total_cost,
        'cost_per_cbo_usd': total_cost / len(results) if results else 0
    }

    # Save token info to a separate file
    token_df = pd.DataFrame([token_info])
    token_df.to_csv('token_usage_cbo.csv', index=False)
    print(f"Informações de uso de tokens salvas em token_usage_cbo.csv")

    # Save results to CSV
    output_file = 'cbos_classificados.csv'
    results_df.to_csv(output_file, index=False, encoding='latin-1', sep=',')
    print(f"Resultados salvos em {output_file}")

    # Also save as Excel file for better handling of special characters
    output_excel = 'cbos_classificados.xlsx'
    results_df.to_excel(output_excel, index=False)
    print(f"Resultados também salvos em {output_excel}")

    # Also save token info to Excel
    token_df.to_excel('token_usage_cbo.xlsx', index=False)
    print(f"Informações de uso de tokens também salvas em token_usage_cbo.xlsx")

    # Display sample results
    print("\nAmostra dos resultados:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(results_df.head())

    print(f"\nTotal de CBOs classificados: {len(results_df)}")

    # Análise dos resultados
    if not results_df.empty:
        # Verificar o nome da coluna para alta rotatividade
        rotatividade_col = 'alta_rotatividade'
        
        # Garantir que estamos procurando pelos valores corretos (pode ser 'sim', 'Sim' ou outros formatos)
        alta_rotatividade = results_df[results_df[rotatividade_col].str.lower() == 'sim'].shape[0]
        pct_alta = (alta_rotatividade / len(results_df)) * 100
        print(f"\nPorcentagem de CBOs com alta rotatividade: {pct_alta:.2f}%")
        
        # Características mais comuns em CBOs de alta rotatividade
        alta_rot_df = results_df[results_df[rotatividade_col].str.lower() == 'sim']
        if not alta_rot_df.empty:
            print("\nCaracterísticas mais comuns em CBOs com alta rotatividade:")
            for col in ['trabalho_extenuante_estressante', 'baixa_escolaridade_exigida', 'sazonalidade', 'alta_exigencia_tecnica', 'salario_mais_elevado', 'interesse_em_reter_talentos']:
                sim_count = alta_rot_df[alta_rot_df[col].str.lower() == 'sim'].shape[0]
                sim_pct = (sim_count / len(alta_rot_df)) * 100
                print(f"{col}: {sim_pct:.2f}% ({sim_count}/{len(alta_rot_df)})")

    # Print summary of token usage and costs
    print("\nResumo de Uso de Tokens e Custos:")
    print(f"Modelo utilizado: {model_name}")
    print(f"Total de CBOs processados: {len(results)}")
    print(f"Total de tokens de entrada: {total_input_tokens}")
    print(f"Total de tokens de saída: {total_output_tokens}")
    print(f"Custo de tokens de entrada (${input_cost_per_million}/M): ${input_cost:.4f}")
    print(f"Custo de tokens de saída (${output_cost_per_million}/M): ${output_cost:.4f}")
    print(f"Custo total: ${total_cost:.4f}")
    print(f"Custo médio por CBO: ${token_info['cost_per_cbo_usd']:.6f}")
else:
    print("Nenhum CBO válido para processar. Verifique o arquivo de entrada e os critérios de filtragem.")
