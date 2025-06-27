# 🍎 Self-Checkout Fruit Detection System: A Deep Dive

Oi Felipe!!!! Seguem algumas dicas

*Este documento detalha o projeto de um sistema de detecção de frutas para terminais de autoatendimento, expandindo a documentação inicial com justificativas técnicas, estratégias de dados e pipeline de produção.*

---

## 1. Desafio de Negócio

O objetivo é desenvolver um **terminal de self-checkout** inteligente capaz de identificar automaticamente as frutas mais comuns — **banana, maçã e uva** — em tempo real. O sistema precisa ser robusto o suficiente para funcionar em múltiplos cenários, como quando o cliente:

- Coloca a fruta solta sobre a balança.
- Pesa as frutas dentro de um saco plástico (transparente ou opaco).
- Segura a fruta com os dedos em frente à câmera.

### 📊 Metas de Performance

O sucesso do projeto será medido por métricas rigorosas de precisão e latência, essenciais para uma experiência de usuário fluida e confiável.


| Métrica            | Valor Alvo | Justificativa                                                                                 |
| --------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| **Precisão Top-1** | ≥ 95%     | Garante que a primeira predição do sistema seja a correta na grande maioria das vezes.      |
| **mAP@0.5**         | ≥ 0.90    | Avalia a capacidade do modelo de localizar corretamente os objetos (IoU > 50%).               |
| **mAP@0.5:0.95**    | ≥ 0.65    | Métrica mais rigorosa que avalia a precisão da localização em diferentes limiares de IoU. |
| **Latência**       | < 100ms    | Essencial para uma resposta em tempo real em hardware embarcado de baixo custo.               |

---

## 2. Escopo e Variações Suportadas

Para garantir que o modelo seja robusto no mundo real, ele será treinado para lidar com uma vasta gama de variações.

#### Subvariedades de Frutas

- **Maçã**: Gala, Fuji, Verde
- **Uva**: Roxa, Verde
- **Banana**: Nanica, Prata, Banana-maçã

#### Condições de Captura de Imagem


| Aspecto            | Variações                                      |
| -------------------- | -------------------------------------------------- |
| **Quantidade**     | `one`, `few` (2-3), `many` (4+)                  |
| **Embalagem**      | `none`, `plastic_bag`                            |
| **Transparência** | `transparent`, `semi_transparent`, `opaque`      |
| **Interação**    | `none`, `held_by_fingers` (oclusão parcial)     |
| **Iluminação**   | `even`, `shadowed`, `glare` (reflexo)            |
| **Ângulo**        | `top_down`, `slightly_tilted`                    |
| **Superfície**    | `black_scale`, `wood_surface`, `plastic_surface` |
| **Qualidade**      | `sharp`, `slightly_blurry`, `noisy`              |

---

## 3. Stack Tecnológico e Justificativas

A escolha da tecnologia foi guiada pela necessidade de performance em tempo real, baixo custo de hardware e um ciclo de desenvolvimento rápido.


| Componente             | Tecnologia                                  |
| ------------------------ | --------------------------------------------- |
| **Modelo Base**        | YOLOv8n (Ultralytics)                       |
| **Geração de Dados** | Gemini / DALL·E 3                          |
| **Anotação**         | Segment Anything (SAM) + YOLO pré-treinado |
| **Augmentation**       | Albumentations                              |
| **Treinamento**        | Google Colab / GPU local (PyTorch)          |
| **Deploy**             | Python + OpenCV + ONNX-Runtime              |

### Por que YOLOv8n?

A escolha do YOLOv8n (nano) não foi acidental. Ele oferece a melhor combinação de velocidade, tamanho e precisão para dispositivos de borda (*edge devices*).


| Critério                  | Vantagem do YOLOv8n                                                      |
| ---------------------------- | -------------------------------------------------------------------------- |
| **Tempo Real**             | >30 FPS em hardware modesto, identificando frutas instantaneamente.      |
| **Tamanho Reduzido**       | ~6 MB, ideal para a memória limitada de um dispositivo embarcado.       |
| **Detecção Multiobjeto** | Localiza múltiplas frutas e tipos de fruta na mesma imagem.             |
| **Pipeline Simples**       | A framework Ultralytics unifica treino, validação e exportação.      |
| **Escalabilidade**         | Permite migrar para modelos maiores (YOLOv8s/m/l) sem alterar o dataset. |

#### Alternativas Consideradas (e por que foram descartadas)


| Modelo                        | Limitação no nosso caso                                                                                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EfficientNet, MobileNetV3** | São modelos de**classificação**, não **detecção**. Não fornecem as caixas delimitadoras (*bounding boxes*) necessárias para localizar múltiplas frutas.                            |
| **Segment Anything (SAM)**    | É uma ferramenta fantástica para**segmentação**, mas é computacionalmente pesada (~200 MB) e lenta para inferência em tempo real. Usaremos no pipeline de anotação, não no deploy. |
| **YOLOv5/7**                  | YOLOv8 possui uma arquitetura mais limpa, é mais preciso e mantido ativamente pela Ultralytics.                                                                                            |

---

## 4. Estratégia de Dados: Geração Sintética

A anotação manual de milhares de imagens é o maior gargalo em projetos de visão computacional. Para contornar isso, adotamos uma **estratégia de dados sintéticos**, que nos permite criar um dataset grande, diverso e perfeitamente balanceado a um custo muito baixo.

O fluxo é o seguinte:

1. **Definir Cenários**: Criar um arquivo CSV com ~500 combinações das variações descritas na Seção 2.
2. **Gerar Prompts**: Um script "Prompt-Factory" converte cada linha do CSV em um prompt detalhado para um modelo de geração de imagem.
3. **Gerar Imagens**: O prompt é enviado ao Gemini ou DALL·E 3 para criar a imagem sintética.
4. **Auto-Anotação**: As imagens geradas são automaticamente anotadas para criar as *bounding boxes*. basta colocar no codigo!
5. Note que nao basta simplesmente colocar a imagem e treinar. Tem que colocar uma anotacao por aimagem. GenAI faz isso .

### A. Schema do CSV (uma linha = uma imagem)


| Coluna             | Valores Possíveis                               |
| -------------------- | -------------------------------------------------- |
| `fruit`            | `banana`, `apple`, `grape`                       |
| `subtype`          | Nanica, Prata / Gala, Fuji / Roxa, Verde         |
| `quantity`         | `one`, `few`, `many`                             |
| `container_type`   | `none`, `plastic_bag`                            |
| `bag_transparency` | `transparent`, `semi_transparent`, `opaque`      |
| `lighting`         | `even`, `shadowed`, `glare`                      |
| `angle`            | `top_down`, `slightly_tilted`                    |
| `interaction`      | `none`, `held_by_fingers`                        |
| `surface_type`     | `black_scale`, `wood_surface`, `plastic_surface` |
| `camera_quality`   | `sharp`, `slightly_blurry`, `noisy`              |

### B. Prompt-Factory

Um LLM recebe a linha do CSV como um JSON e a transforma em uma descrição em inglês fluente.

```prompt
Given a JSON with attributes, write a fluent English prompt describing 
the scene so an image generator can draw it. Mention fruit type + subtype 
+ color, container + transparency, lighting, camera angle, surface, 
fingers if present, quantity, and image quality. Avoid column names.
```

### C. Anotação Rápida com IA

Em vez de desenhar caixas manualmente, usamos um fluxo de IA:

1. **Auto-Segmentar com SAM**: O modelo Segment Anything (SAM) gera máscaras de segmentação de alta precisão para cada fruta na imagem.
2. **Converter para Bounding Box**: A máscara é convertida para uma *bounding box* no formato YOLO.
3. **Bootstrap com YOLO pré-treinado**: Alternativamente, um modelo YOLOv8 treinado no dataset COCO já sabe o que é "banana" e "apple" e pode gerar os rótulos iniciais.
4. **Revisão Manual (10-20%)**: Apenas uma pequena fração do dataset é revisada manualmente para garantir a qualidade e consistência das anotações.

Este processo reduz o esforço de anotação em mais de 90%.

---

## 5. Pipeline de Treinamento e Validação

### Estrutura de Diretórios

O dataset deve seguir a estrutura padrão exigida pelo YOLO.

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### Configuração `data.yaml`

Este arquivo aponta para os dados de treino/validação e define as classes do projeto.

```yaml
path: ./dataset
train: images/train
val: images/val
nc: 3
names: ['banana', 'apple', 'grape']
```

### Augmentation de Imagens

Para aumentar a variabilidade do dataset, aplicamos transformações em tempo real durante o treinamento com `Albumentations`.

```python
import albumentations as A

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.GaussNoise(p=0.15),
    A.MotionBlur(p=0.15),
])
```

### Comando de Treinamento

O treinamento é iniciado com um único comando.

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

### Validação com Dados Reais

Após o treino com dados sintéticos, o modelo é validado contra um pequeno conjunto de **100 fotos reais**, coletadas e anotadas manualmente. Essas imagens reais são então incorporadas ao dataset para um ciclo de refinamento.

---

## 6. Deploy e Ciclo de Vida em Produção

### Export do Modelo

O modelo treinado (um arquivo `.pt` do PyTorch) é exportado para o formato ONNX (Open Neural Network Exchange), que é otimizado para inferência de alta performance em qualquer plataforma.

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

### Inferência e Monitoramento

- **Runtime**: A aplicação final utiliza Python com `ONNX-Runtime` para carregar o modelo e `OpenCV` para capturar os frames da câmera.
- **Fallback**: Se a confiança da predição for muito baixa (ex: < 0.3), o sistema pode solicitar intervenção humana.
- **Logging**: Cada predição é registrada em um log JSON estruturado para análise futura.
  ```json
  {
    "timestamp": "2024-01-01T12:00:00Z",
    "fruit_pred": "banana",
    "confidence": 0.95,
    "bbox": [x, y, w, h]
  }
  ```
- **Retreinamento Contínuo**: Os logs são analisados semanalmente para identificar casos de falha (edge cases). Esses casos são coletados, anotados e incorporados ao dataset para um retreinamento mensal, garantindo que o modelo melhore continuamente.

---

## 7. Cronograma Estimado


| Fase              | Duração | Tarefas Principais                                               |
| ------------------- | ----------- | ------------------------------------------------------------------ |
| **Semanas 1-2**   | 2 semanas | Setup do ambiente, Geração do CSV, Script Prompt-Factory.      |
| **Semanas 3-4**   | 2 semanas | Geração e armazenamento das imagens sintéticas.               |
| **Semanas 5-6**   | 2 semanas | Pipeline de anotação automática e revisão manual.            |
| **Semanas 7-8**   | 2 semanas | Treinamento do modelo inicial e análise de resultados.          |
| **Semanas 9-10**  | 2 semanas | Coleta de dados reais, refinamento e retreinamento.              |
| **Semanas 11-12** | 2 semanas | Export do modelo, testes de integração e documentação final. |

---

## 8. Referências

- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **SAM**: [Segment Anything Model](https://segment-anything.com/)
- **Albumentations**: [Documentation](https://albumentations.ai/)
- **ONNX**: [ONNX Runtime Documentation](https://onnxruntime.ai/)
