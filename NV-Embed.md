# NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models 

코드: https://huggingface.co/nvidia/NV-Embed-v1
논문: https://arxiv.org/pdf/2405.17428

ABSTRACT

Decoder-only LLM-based emvedding모델이 범용적인  임베딩 과제에서 BERT 또는 T5 등을 사용한 임베딩 모델의 성능을 뛰어넘기 시작함
우리가 제안하는 NV-Embed모델은  간단함과 재사용성은 유지하면서도 versatile embedding model로서 LLM의 성능을 높여줌
모델의 구조에서는 pooled embedding을 얻기 위해 latent attention layer를 제안함
representation learning을 향상하기 위해 contrastive learning에서 causal attention mask를 제거함
두 단계의 contrastive instruction-tuning방법론 제안
in-batch negative와 curated hard negative 를 사용해 retrieval 데이터에 instruction을 사용한 contrastive learning을 수행
instruction tuning에서 retrieval 데이터셋이 아닌 데이터를 섞어서 retrieval task와 그것이 아닌 과제들의 성능을 향상함
publicly available data만으로도 MTEB benchmark의 embedding task에서 SOTA를 달성함

1. Introduction
rag는 최신의 정보를 모델의 파라미터 변형 없이 접근할 수 있도록 함
그동안은 양방향 언어모델이 임베딩을 주도해왔고 최근 decoder-only model이 그 성능을 뛰어넘기 시작했지만,
LLM(GPT-4)을 사용해 가상 데이터(synthetic data)를 만들어 LLM fine-tuning에 활용한 기존의 연구흐름은 데이터가 커뮤니티에 공개되지 않았다난 한계가 있음
본 논문에서는 decoder-only model의 임베딩과 retrieval task의 성능을 크게 행상시키는 NV-Embed모델을 제안함
본 연구의 의의
새로운 latent attention layer의 제안
contrastive learning에서 causal attention을 제거하여 solid improvement가 나옴
2. 2단계의 instruciton tuning method제안
[1] in-batch negative와 curated hard negative를 사용해 retrieval 데이터에 instruction을 사용한 contrastive learning을 수행
[2] instruction tuning에서 retrieval 데이터셋이 아닌 데이터를 섞어서 retrieval task와 그것이 아닌 과제들의 성능을 향상함
위 단계에서는 in-batch negative를 사용하지 않음(retrieval task가 아닌 경우에는 in-batch negative가 혼란을 줄 수 있기 때문)
3. MTEB내의 56개의 embedding task에서 69.32점으로 SOTA를 달성함


2. Related work

2.1 Bidirectional Embedding Models
BERT기반의 임베딩은 범용목적의 embedding task를 위해 주로 사용되었음
sentence-BERT, SimCSE 등은 사전학습된 BERT나 T5등에 선별된 비지도된 또는 약하게 지도된 text pair에 의해 대조학습으로 fine-tuning됨 
이후, MS MARCO와 같이 다양한 지도학습된 데이터로 학습됨
모든 SOTA embedding 모델들은 지도학습됨
2.2 Decoder-only LLM-based Embedding Models
Decoder-only LLM은 범용 목적의 임베딩 모델에 대해 성능이 떨어질 것이라고 믿어짐
unidirectional attention은 representation capability를 떨어뜨림
LLM을 scaling하는 것은 고차원의 임베딩을 출력하기 때문에 차원의 저주에 걸릴 것임
Neelakantan et al. (2022)이 GPT-3모델로 임베딩모델을 개발한 것을 시작으로 E5-Mistral, LLM2Vec과 같은 디코더기반의 모델연구가 이루어짐
E5-Mistral의 성공에 이어, SFR-Embedding-Mistral은 retrieval과 non-retrieval데이터셋을 섞어 성능을 향상함
위의 방법은 본 논문에서 제안하는 NV-Embed와 유사하나 아래의 점에서 다름
[1] Mistral-7B모델을 기반으로 publicly available data만을 사용하여 학습
[2] SFR-Embedding-Mistral은 동일 task내에서만 pair를 만듦, 이와 달리 본 논문은 과제의 데이터를 섞어 보다 성능향상
이와 달리 Gecko같은 방법론은 decoder-only LLM으로부터 데이터 pair를 생성하여 bidirectional embedding모델을 distill함
각 쿼리에 대한 passage를 검색하여 데이터 품질을 개선하고 positive와 hard negative를 LLM을 활용하여 relabeling해 데이터의 성능을 높임
GritLM은 text embedding과 generation모델을 하나로 통합

3. Method

3.1 Bidirectional attention
decoder모델의 unidirectional attention은 모델의 표현력을 제한하여, 같은 크기의 BERT나 T5보다 GPT모델은 NLI영역에서 낮은 성능을 보임
이에, 단순히 contrastive learning에서 decoder-only LLM의 causal attention mask를 제거하는 방법을 사용하여 효과를 입증함

3.2 Latent attention layer
bidirectional embedding모델(BERT계열)에서는 mean pooling을, unidirectional embedding 모델(decoder-only model)에서는 <EOS>토큰 임베딩을 사용했지만 두 방법은 mean pooling은 문장의 핵심구 정보를 희석할 수 있다는 한계를 가지고, <EOS> token embedding은 마지막 token의 영향을 많이 받는 recency bias에 영향을 받는다는 한계가 있다.

이에, 본 논문에서는 latent attention layer를 사용하여 범용 임베딩 태스크에 활용할 수 있는 보다 표현력있는 sequence의 pooling을 얻고자 함(transformer의 encoder-decoder attention과 유사)



decoder의 last hidden layer를 query Q ∈ R^{l×d}로 사용함. l은 sequence의 길이, d는 hidden dimension.
latent array K = V ∈ R^{r×d}는 훈련가능한 "사전"으로 사용됨. r은 사전의 latent의 수
cross attention의  결과는 O = softmax(QK^T )V의 수식으로 계산하며 O ∈ R{l×d}차원임

attention다음으로는 GELU activation이 중간에 포함된 MLP레이어를 활용한 두번의 선형변환을 수행함.
본 모델에서 사용한 latent attention layer는 r이 512차원이며 multi head attention에 사용한 head는 8개임
본 논문은 제안한 latent attention layer와 self attention을 비교하여 ablation study에서 지속적인 성능개선을 입증함

3.3 Two-stage instruction tuning

Instrcution tuning은 Instruction에 따르고 RAG를 하는 LLM의 학습을 위해 사용되었음
또한, retriever를 학습하고 다양한 instruction과 ask를 수행할 수 있는 범용 목적의 임베딩 모델을 학습하기 위해 적용되었음
범용 임베딩 모델을 얻기 위해서는 다양한 task들의 특징을 고려해야함
예를 들어, in-batch negative의 활용은 computation을 재사용할 수 있고 B^2로 된 mini batch의 question-passage pair를 question과 그에 상응하는 positive passage B개만 사용하면 되기 때문에 밀집벡터기반 retriever 학습에 효율적임
그러나, in-batch negative를 사용하는건 clustering이나 classification에서 embedding모델에 혼란을 줄 수 있음
왜냐하면 mini-batch의 "passage"는 class에서 오는 것이며 negative가 아니기 때문
이를 고려하여 본 논문에서는 two-stage instruction tuning방법론을 제안함
[1] in-batch negative와 curated hard negative 를 사용해 retrieval 데이터에 instruction을 사용한 contrastive learning을 수행
[2] in-batch negative를 사용하지 않고 retrieval 데이터셋과 아닌 데이터셋을 섞어 instruction tuning을 수행
retrieval task는 다른 task에 비해 어렵기 때문에 retrieval task를 잘 fine-tuning할 수 있는 훈련전략을 사용함

4. Training data
최근 GPT-4 또는 Gemini가 생성한 synthetic data를 활용하지만, 본 논문에서는 embedding task에 대한 모델의 성능을 증명하기 위해 공개데이터를 사용함
qurey-document pair가 있을 때 instructed query는 아래와 같은 템플릿을 따름:


학습과 평가에 대한 최종 임베딩에서 instruction token은 masking함, self-attention과정에서는 서로 영향을 줌
document에서는 instruction prefix를 추가하지않음

4.1 Public Retrieval Datasets
공개데이터를 사용
이러한 데이터는 자체 hard negative가 없기 때문에 encoder-based모델을 따로 학습시켜 데이터셋의 hard negative를 선정

4.2 Public Non-retrieval datasets
retrieval task외의 task에서도 q^+_{inst}, positive document d^+,  hard negative documents d^− _0, ..., d^−_n 을 둠
다양한 classification데이터셋을 사용
중복된 데이터셋을 확인하기 위해 BM25를 사용

5. Experiments
5.1 Experimental details
LoRA, Mistral-7B 사용
attention mask를 causal -> bidirectional로, latent attention layer로 통합(512 latents, 4096 hidden size, 8 multi-head attention)
128 batch로 학습(각 batch는 1개의  positive pair와 7개의 negative pair로 구성된 query가 있음)
평가 단계에서는 이전 연구와 공정한 비교를 위해 512 token limit을 둠

5.2 MTEB Results



평균 69.32로 새로운 sota를 달성.
15가지 retrieval task에 대해서도 59.32로 가장 높은 점수를 달성
다른 모델들과 비교할 때 본 논문은 retrieval task가 다른 task에 비해 더 복잡함을 인지하여 retrieval task를 stage 1에서 먼저 학습하고, stage2에서 다른 task들과 혼합하여 학습해 모든 과제에서의 성능을 향상함
본 논문은 mistral-7B를 base-decoder로 사용하여 불필요한 causal attention mask를 없애고 latent attention layer를 통해 sequence pooling mechanism을 발전시킴

5.3 Ablation study


5.3.1 Causal Attention vs. Bidirectional Attention
bidirectional mask가 causal mask에 비해 지속적으로 높은 성능을 보임

5.3.2 Pooling Methods
mean pooling이 지속적으로 <EOS>-last token embedding에 비해 높은 성능을 보임.
이는 <EOS>-last token embedding이 final token에 많은 영향을 받는 recency bias때문인 것으로 보임
latent attention layer는 모든 과제에서 유익함을 증명
또한, latent attention layer는 decoder-only LLM에서 output embedding representation을 학습하며, output embedding을 평균해서 나타나는 정보 희석을 완화

6. Conclusion
NV-Embed모델은 새로운 구조적 디자인과 두 단계의 훈련단계를 통해 generalist embedding model로서 LLM의 성능을 높임
모델의 구조에서는 표현력있는 pooled embedding을 얻기 위한 latent attention layer와, 불필요한 causal attention mask를 없앰
모델의 학습에서는 두 단계의 contrastive instruction tuning이 순차적으로 retrieval task -> 그외의 task의 성능을 향상시킴
2024.05.24 기준 MTEB에서 sota의 성능 달성
GPT-4 등이 생성한 데이터가 아닌 publicly available data만을 사용하여 성능을 향상하였다는 의의
