# 🍎 Exemplo Prático: Processamento de Frutas com Runnables

Este exemplo demonstra como processar **50 frutas** em paralelo usando LangChain Runnables, mostrando na prática os conceitos explicados no guia principal.

## 🎯 O que o exemplo faz?

O script `aprenda.py` processa uma lista de 50 frutas e gera descrições detalhadas para cada uma, incluindo:
- Descrição (sabor, textura, aparência)
- Origem geográfica
- Benefícios nutricionais
- Curiosidades interessantes
- Melhor época de consumo
- Formas de consumo

## 🚀 Como executar

### 1. Configurar ambiente

```bash
# Instalar dependências
pip install langchain-core langchain-openai openai tqdm python-dotenv tiktoken

# Criar arquivo .env
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
echo "MODEL=gpt-4o-mini" >> .env
```

### 2. Executar o exemplo

```bash
python aprenda.py
```

## 📊 O que você verá

O script executa **5 demonstrações**:

### 1️⃣ **Comparação de Performance**
```
⚡ Comparação: Paralelo vs Sequencial
📝 Amostra: Maçã, Banana, Laranja, Uva, Morango
--------------------------------------------------
🐌 Processamento Sequencial:
🚀 Processamento Paralelo:

📈 Resultados:
  ⏱️  Sequencial: 12.45s
  ⚡ Paralelo: 3.21s
  🎯 Speedup: 3.9x mais rápido
```

### 2️⃣ **Tracking de Tokens**
```
🔍 Processamento com tracking de tokens
--------------------------------------------------
🍓 Processando (com tokens): 100%|████████| 5/5
  📊 Maçã: 156 → 312 tokens
  📊 Banana: 154 → 298 tokens
  📊 Laranja: 158 → 305 tokens
```

### 3️⃣ **Processamento em Escala**
```
🍎 Iniciando processamento de 20 frutas
📦 Tamanho do lote: 5
--------------------------------------------------
🔄 Processando lotes: 100%|████████| 4/4
```

### 4️⃣ **Resultados Salvos**
```
💾 Resultados salvos em: frutas_processadas.json
```

### 5️⃣ **Estatísticas Finais**
```
📊 Estatísticas Finais
==================================================
⏱️  Tempo total: 45.67s
🔤 Tokens entrada: 3,124
🔤 Tokens saída: 6,890
🔤 Total tokens: 10,014
💰 Custo estimado: $0.0459
📈 Tokens/segundo: 219.3
```

## 📁 Arquivos gerados

- **`frutas_processadas.json`**: Contém todas as descrições geradas

Exemplo de resultado:
```json
{
  "nome": "Maçã",
  "descricao": "Fruta crocante e suculenta, com sabor que varia do doce ao levemente ácido...",
  "origem": "Ásia Central, região do Cazaquistão",
  "beneficios": "Rica em fibras, vitamina C e antioxidantes",
  "curiosidade": "Existem mais de 7.500 variedades de maçãs no mundo",
  "melhor_epoca": "Outono e inverno no hemisfério sul",
  "como_consumir": "In natura, sucos, tortas, compotas e saladas"
}
```

## 💡 Conceitos demonstrados

### ✅ **Runnables Básicos**
- Criação de chains simples
- Uso do operador pipe (`|`)
- Templates de prompt

### ✅ **Processamento Paralelo**
- `chain.map()` para múltiplos inputs
- Comparação sequencial vs paralelo
- Controle de batch size

### ✅ **Monitoramento**
- Contagem de tokens
- Cálculo de custos
- Tracking de performance

### ✅ **Boas Práticas**
- Tratamento de erros
- Rate limiting (pausas)
- Salvamento de resultados
- Logging detalhado

## 🔧 Customizações

### Alterar número de frutas
```python
sample_fruits = FRUTAS[:10]  # Processar apenas 10
```

### Mudar batch size
```python
processor = FruitProcessor(batch_size=3)
```

### Usar modelo diferente
```env
MODEL=gpt-4o  # No arquivo .env
```

### Alterar prompt
```python
def create_fruit_prompt_template():
    template = """
    Analise a fruta {fruit_name} focando apenas em:
    - Sabor
    - Benefícios nutricionais
    
    Responda em JSON simples.
    """
    # ...
```

## 💰 Custo Estimado

Para 50 frutas com gpt-4o-mini:
- **Tokens**: ~15.000-20.000 total
- **Custo**: ~$0.10-0.15
- **Tempo**: ~2-3 minutos

## 🎓 Próximos passos

1. Execute o exemplo básico
2. Modifique o prompt para seus dados
3. Teste com diferentes batch sizes
4. Implemente seu próprio caso de uso

Este exemplo serve como base para qualquer processamento em escala com Runnables! 