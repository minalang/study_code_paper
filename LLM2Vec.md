# LLM2Vec

논문 링크: https://arxiv.org/abs/2404.05961

tutorial: https://mcgill-nlp.github.io/llm2vec/tutorial/

앞의 두 단계를 코드적으로 어떻게 변환했는지 보여주고있음 
Official github: https://github.com/McGill-NLP/llm2vec

abstract
최근 decoder only로 구성된 llm이 sota를 달성하고 있지만 풍부한 맥락적 표현을 익혀야하는 임베딩 과제에서의 활용은 느리다. 
이에, 본 논문에서는 decoder-only LLM을 강력한 text encoder로 변환하는 비지도 방법론을 제시한다. 아래와 같은 세 가지 step으로 구성되어있다.
[1] bidirectional attention의 사용
[2] masked next token prediction
[3] unsupervised contrastive learning
 
3개의 LLM(params 1.3B ~ 7B)에 방법론을 평가한 결과 MTEB benchmark에서 encoder-only모델의 점수를 상회. 
Supervised contrastive learning을 적용한 경우 publicly available data에서 좋은 성능을 보임 
 
Introduction 
그동안 text embedding은 token간 풍부한 interaction에 대해 학습해야하기 때문에 causal attention을 통해 미래 단어를 볼 수 없는 decoder-only모델이 활용성이 떨어진다고 판단되었음
하지만 decoder-only모델을 encoder-only모델에 비해 아래와 같은 장점이 있음:
[1] pretrain과정에서 모든 input token을 학습함(encoder-only모델보다 self-efficient함)
[2] LLM에 대해 풍부한 환경(툴, 학습방법 등)이 갖추어져있음
[3] instruction을 따라가는 것에 탁월함(universal text embedding을 만들기에 적합함)
 
따라서, 어떤 pretrained decoder-only model이더라도 text encoder로 변환하는 LLM2VEC방법론을 제안함
 
MTEB benchmark에서 unsupervised model에 대해 sota달성
Supervised contrastive learning과 결합할 겨우 publicly available한 data만으로 학습시켰을 때 sota달성
 
LLM2Vec 


Three simple ingredients 
1) Enabling bidirectional attention(Bi) 
Causal attention의 mask를 all-ones matix로 대체하여 각 토큰들이 sequence의 다른 토큰들에 접근할 수 있도록 해 bidirectional LLM으로 변환함
하지만, 이것이 왜 더 나은 sequence representation을 만드는지 명백하게 밝혀지지 않았기 때문에 미숙한 접근은 성능을 떨어지게 할 수 있다. 실험에서도, 단순히 bidirecional attention을 적용하는 것만으로는 성능이 떨어지기도 하였음
다만 구현은 간단함
 
2) Masked next token prediction(MNTP) 
모델이 bidirectional attention을 인지하게 하기 위해 MNTP를 사용
MNTP는 next token prediction과 masked language modeling을 결합한 학습방법임
임의의 sequence가 있다고 가정할 때 몇몇의 token들을 임의로 가린 뒤 과거와 미래의 토큰을 통해 mask를 예측하게 함
i번째 masked token을 맞추기 위해서는 i-1번째의 loss based logit을 계산함
 

3) Unsupervised contrastive learning(simCSE) 
앞의 두 단계가 decoder-only모델을 word-level의 encoder로 변환시켜주기는 하지만 sequence representation으로는 충분하지 않음
Decoder-only모델은 전체적인 sequence의 특징을 잡아내도록 훈련되지 않았기 때문
이에, unsupervised simCSE를 활용하여 같은 문장의 hidden representation에 random dropout mask를 사용하고 noise를 섞은 문장을 positive pair로 하여 가깝게 학습하고, 배치 내의 다른 문장을 negative pair로 하여 유사도를 줄이는 방향으로 학습진행
모든 sequence에 적용할 수 있다는 장점. sequence representation을 얻기 위해 token representation에 pooling을 적용함
 
Transforming decoder-only LLMs with LLM2Vec 
Params 1.3B ~ 7B까지의 모델을 실험에 활용 
Sheared-LLaMA-1.3B (S-LLaMA-1.3B, Xia et al., 2023), 
Llama-2-7B-chat (LLaMA-2-7B, Touvron et al., 2023), 
Mistral-7B-Instruct-v0.2 (Mistral-7B, Jiang et al., 2023)
 
Training data 
MNTP와 SimCSE는 English Wikipedia활용
-> 실험 대상 모델들의 사전학습과정에서 이미 학습한 데이터이기 때문에 새로운 지식을 배우지는 못했다고 판단함
 
Masked next token prediction 
Underscore(_)를 mask token으로 사용함(masking에 사용한 special token이 없기 때문)
LoRA를 통한 fine-tuning
학습setting 세부:
trained for 1000 steps with a batch size of 32 on a single 80GB A100 GPU. For 7B models, this training takes only 90 minutes. 
 
Unsupervised contrastive learning 
MNTP LoRA 가중치와 base model의 가중치를 병합하고, simCSE학습을 시작하기 전 새로 학습한 LoRA파라미터를 초기화 하여 이전에 step에서 배운 지식을 유지하도록 함
MNTP와 동일하게 1000 step 학습, 
7b모델의 경우 80G 단일 A100 gpu, batch size 128, 2.5시간 학습
 
LLM2Vec-transformed models are strong unsupervised text embedders 
Evaluation on word-level tasks 
 


Evaluation on sequence-level tasks 


 
How does LLM2Vec affect a model? 
 
LLM2Vec helps models to capture information from future tokens



LM2Vec 변환한 모델이 미래 토큰 정보를 잘 통합하는지 분석하기 위해 springer et al(2024)의 방법을 차용하여 다른 문장이지만 같은 접두사를 가진 문장의 유사도를 어떻게 판단하는지 실험함
실험결과 MNTP로의 학습은 positive와 negative examples를 명백하게 구분하기에 충분한 학습 방법임을 관찰함
 

Why does bidirectional attention without training work for Mistral models? 

본 연구에서는 mistral-7b모델의 경우 심지어 학습을 하지 않은 경우에도 bidirectional attention을 활성화하는 것이 효과가 있다는 실험결과가 나왔음
 



실험결과, bidirectional attention을 활성화하면 S-LLaMA-1.3B나 LLaMA-2-7B의 경우 모든 layer와 token position에서 낮은 코사인 유사도를 보이며 모호한 효과를 보이지만 mistral의 경우 모든 표현들의 전반적으로 높은 코사인유사도를 보임
이에, 연구자들은 mistral모델이 prefix language modeling과 같은 bidirectional attention을 사용한 사전학습방법을 사용했을 것이라고 추측함 

Combining LLM2Vec with supervised contrastive learning 

LLM2Vec leads to strong performance on the MTEB leaderboard

LLM2Vec과 supervised contrastive learning의 결합하는 모델의 평가 결과



hard negative 와 in-batch negative를 통해 학습 + LoRA fine-tuning
세 가지 모델 모두, LLM2Vec으로 모델을 변환하면 강력한 Unidirectional + 가중 평균 기준선보다 성능이 향상됨 
supervised learning에 대해 unsupervised SimCSE를 수행하는 것은 그다지 중요하지 않으며, LLaMA-2-7B와 Mistral-7B의 경우 LLM2Vec의 MNTP 단계만 수행하는 것에 비해 성능이 약간 떨어짐 
그러나,  MNTP와 SimCSE를 가진 LLM2Vec은 훨씬 더 표본 효율적이며 특히, Mistral-7B (SimCSE 없는 LLM2Vec)는 publicly available data만으로 훈련된 모델 중에서 최신 기록을 세움

5.2 LLM2Vec leads to more sample-efficient training



LLM2Vec으로 변환한 모델(backbone으로 사용한 세 모델 모두)은 훈련 중 이른 기간에 좋은 성과를 냄


Conclusion
본 논문에서는 decoder-only model을 universal text embedder로 변환하는 비감독 방법인 LLM2Vec을 제안
word 및 sequence task에 대해 평가를 진행하고, unsupervised와 supervised 세팅에서의 효율성을 입증함
Mistral-7B에 방법을 적용한 것은 unsupervised task에 대한 MTEB benchmark에서 SOTA를 달성
 LLM2Vec은 간단함과 효율성으로 low-resource환경에서의 유망한 해결책이 될 것임
