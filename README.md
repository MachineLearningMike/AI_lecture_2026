# AI Lecture 2026

Deep-learning-focused AI course.

## Gravity

> Begin delivery. Advance toward understanding -- fast.

File numbers: **4-digit**, step **10**.

**Ch03 Exploit LLM** covers the full amateur→professional API ladder
(language → data → tools → agents → systems → production), before training-from-scratch.

Map: [_curriculum.py](_curriculum.py).

## Chapters

### Ch01: Contract and gravity

- Opener: 0100_Ch01_Contract.py
- Attempt: Say Vision / Mission / Strategy / Attitude in your own words.

| Tag | File | Note |
|-----|------|------|
| Core | 0110_lecture_strategy.py | Vision, mission, gravity, contract |
| Core | 0120_intelligence.py | Working definition of intelligence |

### Ch02: Why deep learning

- Opener: 0130_Ch02_Why_Deep_Learning.py
- Attempt: Pick one capability x one generalization level; argue strong vs fragile.

| Tag | File | Note |
|-----|------|------|
| Core | 0140_DL_history.py | Historical achievements and enabling ideas |
| Core | 0150_DL_future.py | Capability ladder x generalization axis |
| Core | 0160_continuous_end_to_end.py | Language-free continuous representations |

### Ch03: Exploit LLM -- API ladder amateur to professional

- Opener: 0170_Ch03_Exploit_LLM.py
- Attempt: Map one real work task onto the ladder level you will ship this week.

| Tag | File | Note |
|-----|------|------|
| Core | `0172_Opportunity.py` | Why exploit LLMs now -- local PCs vs hyperscale growth |
| Core | `0173_train_GPT.py` | How ChatGPT-class models are trained (pre/post) |
| Core | `0180_exploit_stage1_talk.py` | Stage I: talk -- NL, instructions, context, few-shot |
| Core | `0190_exploit_stage2_data.py` | Stage II: data processor -- text/JSON/multimodal |
| Core | `0200_exploit_stage3_compute.py` | Stage III: tools, Python, DB, your ML models |
| Core | `0210_exploit_stage4_systems.py` | Stage IV: RAG, state, agents, batch pipelines |
| Core | `0220_exploit_eval_production.py` | Evaluation, observability, production architecture |
| Core | `0230_exploit_deep_progression.py` | Deep progression + engineering principle |
| Core | `0232_Vision_Language_Model.py` | VLMs -- vision tokens, projectors, CLIP to LLaVA to agents |
| Core | `0235_agentic_AI.py` | What an AI agent is -- tools, ReAct, memory, planning, RL |
| Core | `0236_agentic_AI_example.py` | Build a data-analyst agent with an existing LLM -- no training |

### Ch04: First training delivery -- see a model train

- Opener: 0240_Ch04_First_Delivery.py
- Attempt: Run the training demo; change one knob; explain what happened out loud.

| Tag | File | Note |
|-----|------|------|
| Core | 0250_AI_fun.py | Safe playground -- confidence first |
| Core | 0260_numpy_torch.py | ndarray vs Tensor |
| Core | 0270_torch_training.py | Lean training loop (+ notebook) |
| Lab | 0280_model.py | Small end-to-end model toy |

### Ch05: Learning as binding

- Opener: 0290_Ch05_Learning_As_Binding.py
- Attempt: Reclassify ImageNet, GPT pretrain, and RLHF in binding terms.

| Tag | File | Note |
|-----|------|------|
| Core | 0300_SL_UL_RL.py | SL / UL / RL + binding climax |
| Core | 0310_regression_classification.py | Two kinds of prediction questions |

### Ch06: Data and distribution

- Opener: 0320_Ch06_Data_And_Distribution.py
- Attempt: Describe one real dataset's shift risk using 150_DL_future levels.

| Tag | File | Note |
|-----|------|------|
| Core | 0330_distribution.py | Why distributions -- business gravity |
| Core | 0340_normalize_data.py | Scale the data (not activation BN/LN) |
| Core | 0350_pipeline_feature_engineering.py | Pipeline and features |

### Ch07: Evaluation and field use

- Opener: 0360_Ch07_Evaluation.py
- Attempt: For a class-imbalanced task, argue why accuracy alone lies.

| Tag | File | Note |
|-----|------|------|
| Core | 0370_F1_Score.py | Confusion matrix, precision, recall, F1 |
| Core | 0380_train_validate_test_deploy.py | Train / validate / test / deploy |

### Ch08: Loss -- scoring the binding

- Opener: 0390_Ch08_Loss_Objectives.py
- Attempt: Pick MSE vs MAE vs CE for one toy task and defend the choice.

| Tag | File | Note |
|-----|------|------|
| Core | 0400_loss_functions.py | What a loss is |
| Core | 0410_Loss_Mean_Squared_Error.py | MSE |
| Core | 0420_Loss_Mean_Absolute_Error.py | MAE |
| Core | 0430_Loss_Huber_Loss.py | Huber |
| Core | 0440_Loss_Binary_Cross_Entropy.py | BCE |
| Core | 0450_Loss_Categorical_Cross_Entropy.py | Categorical CE |
| Lab | 0460_Loss_Kullback_Leibler_Divergence.py | KL |
| Core | 0470_maximum_likelihood.py | MLE bridge to MSE / MAE |

### Ch09: Optimization -- how parameters move

- Opener: 0480_Ch09_Optimization.py
- Attempt: Explain minibatch noise in one minute without equations.

| Tag | File | Note |
|-----|------|------|
| Core | 0490_derivative.py | Derivatives / partials |
| Core | 0500_fastest_ascent.py | Gradient direction |
| Core | 0510_rolling_ball.py | Descent physics intuition |
| Core | 0520_stochastic_ascent.py | Noisy steps still climb |
| Core | 0530_momentum.py | Momentum under noise |
| Core | 0540_SGD_concept.py | SGD concept |
| Core | 0550_SGD_training.py | Full-batch vs minibatch |
| Lab | 0560_back_propagation.py | Chain rule + autograd |
| Core | 0570_AdamW_and_LR_schedules.py | Adam / AdamW + LR schedules |

### Ch10: Probability toolkit (just-in-time)

- Opener: 0580_Ch10_Probability_Toolkit.py
- Attempt: State LLN vs CLT in one sentence each, with an example.

| Tag | File | Note |
|-----|------|------|
| Core | 0590_gaussian.py | Gaussian basics |
| Core | 0600_law_of_large_numbers.py | LLN |
| Core | 0610_central_limit_theorem.py | CLT |
| Core | 0620_probability_vs_statistics.py | Two directions |
| Core | 0630_Bayesian_theorem.py | Bayes update |
| Core | 0640_exponential.py | exp / log in ML |
| Enrichment | 0650_average_of_random.py | Average of r.v.s |
| Enrichment | 0660_product_of_random.py | Product of r.v.s |
| Enrichment | 0670_stress_test.py | Misuse of averaging intuition |

### Ch11: What a net is -- function class and bias

- Opener: 0680_Ch11_Neural_Function_Class.py
- Attempt: Why stacking linear layers without activations collapses.

| Tag | File | Note |
|-----|------|------|
| Core | 0690_inductive_bias.py | Inductive bias |
| Core | 0700_neural_non_linearity.py | Nonlinearity / manifolds |
| Core | 0710_dropout.py | Dropout |
| Core | 0720_normalize_activations.py | BatchNorm vs LayerNorm |
| TODO | *(TODO)* | CNN inductive bias |
| TODO | *(TODO)* | Initialization + residual networks |
| TODO | *(TODO)* | Overfitting / underfitting visual lecture |

### Ch12: Discriminative vs generative

- Opener: 0730_Ch12_Discriminative_Generative.py
- Attempt: Name one task that needs P(x) and one that only needs P(y|x).

| Tag | File | Note |
|-----|------|------|
| Core | 0740_discriminative_generative.py | Conceptual contrast |
| Lab | 0750_discriminative_toys.py | Discriminative toys |
| Lab | 0760_generative_toys.py | Generative toys |
| TODO | *(TODO)* | Diffusion intuition |

### Ch13: Representation -- tokens become geometry

- Opener: 0770_Ch13_Representation.py
- Attempt: Explain why a word vector is useful without listing Word2Vec math.

| Tag | File | Note |
|-----|------|------|
| Core | 0780_Embedding.py | Embedding tables |
| Core | 0790_word_vector.py | Word2Vec intuition |
| Core | 0800_linear_transformation.py | Linear maps as matrices |
| Core | 0810_dot_inner_scalar_product.py | Dot product / similarity |
| Core | 0820_softmax.py | Softmax |

### Ch14: Transformer -- deliver, then unpack

- Opener: 0830_Ch14_Transformer.py
- Attempt: Run or read the off-the-shelf path; then explain one attention head.

| Tag | File | Note |
|-----|------|------|
| Core | 0840_off_the_shelf_transformer.py | Call library blocks first |
| Core | 0850_transformer.py | Self-attention intuition |
| Core | 0860_QKV_attention.py | Q, K, V meanings |
| Core | 0870_Wq_Wk_Wv.py | Learned projections |
| Core | 0880_positional_encoding.py | Positions |
| Core | 0890_chain_of_adds.py | Residual chain |
| Core | 0900_model_dimension.py | Width vs sequence length |
| Lab | 0910_transformer_code.py | Shapes from scratch |
| Core | 0920_Transformer_inductive_bias.py | Transformer priors |
| Lab | 0930_machine_translation.py | Tiny NMT |
| TODO | *(TODO)* | RNN / LSTM bridge -- problem attention solved |

### Ch15: LLM internals -- BERT / GPT mechanisms

- Opener: 0940_Ch15_LLMs.py
- Attempt: Contrast masked pretraining vs next-token pretraining in binding words.

| Tag | File | Note |
|-----|------|------|
| Core | 0950_BERT.py | Encoder pretraining + survey |
| Core | 0960_GPT.py | Decoder LLMs, SFT path |
| TODO | *(TODO)* | RLHF / preference as external binding on top of pretrain |

### Ch16: Variants (enrichment)

- Opener: 0970_Ch16_Variants.py
- Attempt: Name one reason global attention is expensive on images/video.

| Tag | File | Note |
|-----|------|------|
| Enrichment | 0980_video_transformer.py | Video tokens / attention |
| Enrichment | 0990_swin_transformer.py | Shifted windows |

### Parallel: Speaking clinic (parallel track)

- Opener: 1000_ChParallel_Speaking.py
- Attempt: Answer one interview question twice: native clarity, then your voice.

| Tag | File | Note |
|-----|------|------|
| Core | 1010_Asian_interview.py | Native vs Asian English bank |

## Minimal core path

- 0100_Ch01_Contract.py
- 0110_lecture_strategy.py
- 0120_intelligence.py
- 0130_Ch02_Why_Deep_Learning.py
- 0140_DL_history.py
- 0150_DL_future.py
- 0160_continuous_end_to_end.py
- 0170_Ch03_Exploit_LLM.py
- 0172_Opportunity.py
- 0173_train_GPT.py
- 0180_exploit_stage1_talk.py
- 0190_exploit_stage2_data.py
- 0200_exploit_stage3_compute.py
- 0210_exploit_stage4_systems.py
- 0220_exploit_eval_production.py
- 0230_exploit_deep_progression.py
- 0232_Vision_Language_Model.py
- 0235_agentic_AI.py
- 0236_agentic_AI_example.py
- 0240_Ch04_First_Delivery.py
- 0250_AI_fun.py
- 0260_numpy_torch.py
- 0270_torch_training.py
- 0290_Ch05_Learning_As_Binding.py
- 0300_SL_UL_RL.py
- 0360_Ch07_Evaluation.py
- 0380_train_validate_test_deploy.py
- 0480_Ch09_Optimization.py
- 0570_AdamW_and_LR_schedules.py
- 0830_Ch14_Transformer.py
- 0840_off_the_shelf_transformer.py
- 0940_Ch15_LLMs.py
- 0960_GPT.py

## Notes

- Ch03 = **Exploit LLM** (API usage ladder).
- Ch15 = LLM **internals** (BERT/GPT mechanisms).
- Ch04 = first **training** delivery.

