# 架构图生成规格

使用内置图像生成模型，以原始十张 PNG 和当前代码为依据。统一要求：BERT 分层、立体 token/tensor、金紫色编码器、青色投影、紫色 Q/K、红色条件路径、白底深色技术标签，逐模型绘制；检查后针对维度和残差连接迭代修正。

## task1-baseline

参考图：task1-1.png

生成计算规格：BERT last-four average [L,768] -> Dense [L,713,128] -> split Q/K [L,713,64] -> RoPE -> gather target start/end -> dot product / sqrt(64) -> 713 Frame logits.

## task1-target-aware

参考图：task1-2.png

生成计算规格：BERT last layer [L,768] -> gather CLS, target start, target end, target mean -> concat [3072] -> LayerNorm Dropout Linear -> [713] Frame logits. Show marker input as optional. Baseline ensemble is a separate optional branch weight .375 baseline and .625 target-aware.

## task1-prototype

参考图：task1-3.png

生成计算规格：Definition text -> frozen BERT pooling -> fixed prototype bank [713,768]. Sentence BERT -> CLS/start/end/mean concat [3072] -> LayerNorm Dropout Linear GELU LayerNorm [768] -> normalized context. Cosine dot with normalized bank, learnable scale -> Frame logits. Postprocessing separate: subtract gamma=2.7 from seen logits, equal blend with baseline.

## task2-baseline

参考图：task2-1.png

生成计算规格：Target marked sentence -> BERT last layer [L,768] -> Dense [L,128] -> Q/K [L,64] -> RoPE -> QK transpose / sqrt(64) [L,L]. Training valid span mask + multilabel loss + FGM. Inference threshold and Safe Decode outside network.

## task2-target-gate

参考图：task2-2.png

生成计算规格：BERT last layer [L,768]. Target mean [768] -> linear tanh [768] -> broadcast. Concat token and broadcast [L,1536] -> sigmoid gate [L,768]; hidden + dropout(gate * target update) -> LayerNorm -> Dense Q/K, RoPE, GlobalPointer span matrix [L,L].

## task2-content

参考图：task2-3.png

生成计算规格：BERT [L,768] -> base GlobalPointer [L,L]. Parallel: linear tanh token projection [L,32] -> prefix sums -> all span mean [L,L,32]. Target pool -> separate linear tanh [32]. Dot span mean with target / sqrt(32) -> content scores [L,L]. Base scores + learnable scalar times content scores -> Safe Decode. Do not invent concat matching.

## task2-boundary

参考图：task2-4.png

生成计算规格：BERT last layer -> GlobalPointer span matrix [L,L]. Parallel linear start head [L] and end head [L]. Training span loss + .1 boundary loss. Inference span(i,j)+.34*(start(i)+end(j)) -> threshold -> Safe Decode. Show two distinct train and inference paths.

## task2-definition

参考图：task2-5.png

生成计算规格：Target-marked sentence + frame definition + FE names -> joint BERT. Sentence hidden [L,768] -> base GlobalPointer [L,L]. FE name pooling -> R role queries -> role-specific start and end compatibility with sentence tokens -> [R,L,L] additive residual to base scores. Max over roles -> span matrix -> threshold and Safe Decode. Oracle Frame research setting label.

## task3-interaction

参考图：task3-1.png

生成计算规格：BERT last-four mean [L,768]. Target start/end/mean concat [2304] -> projection [256]. Argument start/end/mean concat [2304] -> projection [256]. concat argument,target,product,absolute difference,distance [32],width [16] = [1072]. MLP -> role query [256] dot shared role bank [1009,256] + bias -> role logits. C local CE; C2 local CE + .2 global CE.

## task3-verifier

参考图：task3-2.png

生成计算规格：Marker sentence -> BERT last-four average. CLS [768], target marker mean [768], argument marker mean [768], abs difference [768], product [768] -> concat [3840] -> Linear GELU LayerNorm Dropout -> query [512]. Separate Task1 frame probs [713] times learnable embeddings [713,128] -> linear [512]. concat query/frame [1024] -> sigmoid gate [512]; LayerNorm(query+gate*frame) -> validity linear sigmoid and role linear [1009]. Conservative rank rejection as separate postprocess.


