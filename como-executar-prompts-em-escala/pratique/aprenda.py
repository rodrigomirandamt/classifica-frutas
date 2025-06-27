#!/usr/bin/env python3
"""
Exemplo Prático: Processamento em Escala com LangChain Runnables
================================================================

Este script demonstra como processar 50 frutas em paralelo usando Runnables,
gerando descrições personalizadas para cada uma.

Autor: Exemplo para TCC Felipe Horta
"""

import os
import time
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import tiktoken
from tqdm import tqdm

# Carregar variáveis de ambiente
load_dotenv()

# Lista de 50 frutas para processar
FRUTAS = [
    "Maçã", "Banana", "Laranja", "Uva", "Morango", "Abacaxi", "Manga", "Pêra", "Melancia", "Melão",
    "Kiwi", "Limão", "Abacate", "Cereja", "Ameixa", "Pêssego", "Coco", "Mamão", "Goiaba", "Maracujá",
    "Figo", "Romã", "Caqui", "Nectarina", "Framboesa", "Mirtilo", "Amora", "Cranberry", "Physalis", "Carambola",
    "Pitaya", "Lichia", "Rambutan", "Durião", "Jackfruit", "Açaí", "Cupuaçu", "Caju", "Jabuticaba", "Pitanga",
    "Uvaia", "Grumixama", "Cambuci", "Feijoa", "Atemoia", "Fruta-do-conde", "Graviola", "Sapoti", "Jenipapo", "Buriti"
]

def setup_openai():
    """Configura o modelo OpenAI"""
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL", "gpt-4o-mini")  # Modelo mais barato para exemplo
    
    if not api_key:
        raise ValueError("⚠️  OPENAI_API_KEY não encontrada no arquivo .env")
    
    model = ChatOpenAI(
        model=model_name, 
        temperature=0.7,  # Um pouco de criatividade para descrições
        api_key=api_key
    )
    
    print(f"✅ Modelo configurado: {model_name}")
    return model, model_name

def create_fruit_prompt_template():
    """Cria template para descrição de frutas"""
    template = """
    Descreva a fruta {fruit_name} de forma interessante e educativa.
    
    Forneça as informações em formato JSON:
    {{
        "nome": "{fruit_name}",
        "descricao": "Descrição detalhada da fruta (sabor, textura, aparência)",
        "origem": "Região/país de origem",
        "beneficios": "Principais benefícios nutricionais",
        "curiosidade": "Uma curiosidade interessante sobre a fruta",
        "melhor_epoca": "Melhor época para consumo",
        "como_consumir": "Formas populares de consumo"
    }}
    
    Seja criativo mas factual!
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em frutas e nutrição. Forneça informações precisas e interessantes."),
        ("human", template)
    ])

def count_tokens(text, model="gpt-4o-mini"):
    """Conta tokens usando tiktoken"""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))

def calculate_costs(input_tokens, output_tokens, model_name="gpt-4o-mini"):
    """Calcula custos baseado no modelo"""
    pricing = {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},  # Preços do gpt-4o-mini
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-4": {"input": 30.0, "output": 60.0}
    }
    
    rates = pricing.get(model_name, {"input": 0.15, "output": 0.6})
    
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

class FruitProcessor:
    """Processador de frutas usando Runnables"""
    
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.model, self.model_name = setup_openai()
        self.setup_chains()
        
        # Estatísticas
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.start_time = None
        
    def setup_chains(self):
        """Configura as chains de processamento"""
        prompt_template = create_fruit_prompt_template()
        output_parser = StrOutputParser()
        
        # Chain básica
        self.chain = prompt_template | self.model | output_parser
        
        # Chain para processamento paralelo
        self.map_chain = self.chain.map()
        
        # Chain com controle de tokens
        self.token_chain = self.create_token_aware_chain()
        
        print("✅ Chains configuradas")
    
    def create_token_aware_chain(self):
        """Cria chain que monitora tokens"""
        def process_with_tokens(inputs):
            # Contar tokens de entrada
            input_text = str(inputs)
            input_tokens = count_tokens(input_text, self.model_name)
            
            # Processar com a chain
            result = self.chain.invoke(inputs)
            
            # Contar tokens de saída
            output_tokens = count_tokens(result, self.model_name)
            
            # Atualizar totais
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            return {
                "result": result,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "fruit_name": inputs["fruit_name"]
            }
        
        return RunnableLambda(process_with_tokens)
    
    def process_fruit_list(self, fruits):
        """Processa lista de frutas em paralelo"""
        self.start_time = time.time()
        
        print(f"\n🍎 Iniciando processamento de {len(fruits)} frutas")
        print(f"📦 Tamanho do lote: {self.batch_size}")
        print("-" * 50)
        
        # Preparar inputs
        inputs = [{"fruit_name": fruit} for fruit in fruits]
        
        # Processar em lotes
        results = []
        total_batches = len(inputs) // self.batch_size + (1 if len(inputs) % self.batch_size else 0)
        
        for i in tqdm(range(0, len(inputs), self.batch_size), 
                     desc="🔄 Processando lotes", 
                     total=total_batches):
            batch = inputs[i:i + self.batch_size]
            
            # Usar map() para processamento paralelo
            batch_results = self.map_chain.invoke(batch)
            results.extend(batch_results)
            
            # Pequena pausa para não sobrecarregar a API
            time.sleep(0.5)
        
        return results
    
    def process_with_token_tracking(self, fruits):
        """Processa com tracking detalhado de tokens"""
        print(f"\n🔍 Processamento com tracking de tokens")
        print("-" * 50)
        
        results = []
        
        for fruit in tqdm(fruits[:5], desc="🍓 Processando (com tokens)"):  # Apenas 5 para exemplo
            result = self.token_chain.invoke({"fruit_name": fruit})
            results.append(result)
            
            print(f"  📊 {fruit}: {result['input_tokens']} → {result['output_tokens']} tokens")
        
        return results
    
    def demonstrate_parallel_vs_sequential(self, fruits_sample):
        """Demonstra diferença entre processamento paralelo e sequencial"""
        sample = fruits_sample[:5]  # Pequena amostra
        
        print(f"\n⚡ Comparação: Paralelo vs Sequencial")
        print(f"📝 Amostra: {', '.join(sample)}")
        print("-" * 50)
        
        # Processamento sequencial
        print("🐌 Processamento Sequencial:")
        start_seq = time.time()
        sequential_results = []
        for fruit in sample:
            result = self.chain.invoke({"fruit_name": fruit})
            sequential_results.append(result)
        sequential_time = time.time() - start_seq
        
        # Processamento paralelo
        print("\n🚀 Processamento Paralelo:")
        start_par = time.time()
        inputs = [{"fruit_name": fruit} for fruit in sample]
        parallel_results = self.map_chain.invoke(inputs)
        parallel_time = time.time() - start_par
        
        # Resultados
        print(f"\n📈 Resultados:")
        print(f"  ⏱️  Sequencial: {sequential_time:.2f}s")
        print(f"  ⚡ Paralelo: {parallel_time:.2f}s")
        print(f"  🎯 Speedup: {sequential_time/parallel_time:.1f}x mais rápido")
        
        return sequential_results, parallel_results
    
    def save_results(self, results, filename="frutas_processadas.json"):
        """Salva resultados em arquivo JSON"""
        parsed_results = []
        
        for result in results:
            try:
                # Tentar fazer parse do JSON
                if isinstance(result, dict) and 'result' in result:
                    content = result['result']
                else:
                    content = result
                
                # Limpar possível markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(content)
                parsed_results.append(parsed)
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Erro ao fazer parse de resultado: {e}")
                parsed_results.append({"error": "Parse error", "raw_content": str(result)})
        
        # Salvar arquivo
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(parsed_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Resultados salvos em: {filename}")
        return parsed_results
    
    def print_statistics(self):
        """Imprime estatísticas finais"""
        if self.start_time:
            total_time = time.time() - self.start_time
            costs = calculate_costs(
                self.total_input_tokens, 
                self.total_output_tokens, 
                self.model_name
            )
            
            print(f"\n📊 Estatísticas Finais")
            print("=" * 50)
            print(f"⏱️  Tempo total: {total_time:.2f}s")
            print(f"🔤 Tokens entrada: {self.total_input_tokens:,}")
            print(f"🔤 Tokens saída: {self.total_output_tokens:,}")
            print(f"🔤 Total tokens: {(self.total_input_tokens + self.total_output_tokens):,}")
            print(f"💰 Custo estimado: ${costs['total_cost']:.4f}")
            print(f"📈 Tokens/segundo: {(self.total_input_tokens + self.total_output_tokens)/total_time:.1f}")

def main():
    """Função principal - demonstra diferentes usos dos Runnables"""
    print("🍎 Exemplo Prático: Processamento de Frutas com Runnables")
    print("=" * 60)
    
    # Inicializar processador
    processor = FruitProcessor(batch_size=5)
    
    # 1. Demonstração: Paralelo vs Sequencial
    print("\n1️⃣  DEMONSTRAÇÃO: Diferença de Performance")
    processor.demonstrate_parallel_vs_sequential(FRUTAS)
    
    # 2. Processamento com tracking de tokens
    print("\n2️⃣  TRACKING DE TOKENS")
    token_results = processor.process_with_token_tracking(FRUTAS)
    
    # 3. Processamento completo das primeiras 20 frutas
    print("\n3️⃣  PROCESSAMENTO EM ESCALA")
    sample_fruits = FRUTAS[:20]  # Usar apenas 20 para economizar tokens
    results = processor.process_fruit_list(sample_fruits)
    
    # 4. Salvar resultados
    print("\n4️⃣  SALVANDO RESULTADOS")
    parsed_results = processor.save_results(results)
    
    # 5. Mostrar exemplo de resultado
    if parsed_results:
        print(f"\n5️⃣  EXEMPLO DE RESULTADO")
        print("-" * 30)
        first_result = parsed_results[0]
        if isinstance(first_result, dict) and 'nome' in first_result:
            print(f"🍎 Fruta: {first_result['nome']}")
            print(f"📝 Descrição: {first_result.get('descricao', 'N/A')[:100]}...")
            print(f"🌍 Origem: {first_result.get('origem', 'N/A')}")
            print(f"💡 Curiosidade: {first_result.get('curiosidade', 'N/A')[:100]}...")
    
    # 6. Estatísticas finais
    processor.print_statistics()
    
    print(f"\n✅ Processamento concluído!")
    print(f"📁 Verifique o arquivo 'frutas_processadas.json' para ver todos os resultados")

if __name__ == "__main__":
    # Verificar se arquivo .env existe
    if not os.path.exists('.env'):
        print("⚠️  Arquivo .env não encontrado!")
        print("📝 Crie um arquivo .env com:")
        print("OPENAI_API_KEY=sua_chave_aqui")
        print("MODEL=gpt-4o-mini")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Processamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc() 