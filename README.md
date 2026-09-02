# AI Lecture 2026

Deep-learning-focused AI course.

## Gravity

> Begin delivery. Advance toward understanding -- fast.

Consecutive lecture numbers differ by **10**, zero-padded to **4 digits**
(`0100`, `0110`, …) so sorting stays correct past `0999`.

**Ch03 LLM usage** comes right after the DL story, before training-from-scratch.

Map: [`_curriculum.py`](_curriculum.py). Critique: [`comment.txt`](comment.txt).

## Chapters

### Ch01: Contract and gravity

- Opener: `0100_Ch01_Contract.py`
- Attempt: Say Vision / Mission / Strategy / Attitude in your own words.

| Tag | File | Note |
|-----|------|------|
| Core | `0110_lecture_strategy.py` | Vision, mission, gravity, contract |
| Core | `0120_intelligence.py` | Working definition of intelligence |

### Ch02: Why deep learning

- Opener: `0130_Ch02_Why_Deep_Learning.py`
- Attempt: Pick one capability x one generalization level; argue strong vs fragile.

| Tag | File | Note |
|-----|------|------|
| Core | `0140_DL_history.py` | Historical achievements and enabling ideas |
| Core | `0150_DL_future.py` | Capability ladder x generalization axis |
| Core | `0160_continuous_end_to_end.py` | Language-free continuous representations |

### Ch03: LLM usage -- deliver with models first

- Opener: `0170_Ch03_LLM_Usage.py`
- Attempt: Complete one real task with an LLM: prompt, verify, cite or refuse.

| Tag | File | Note |
|-----|------|------|
| Core | `0180_prompting_and_verification.py` | Prompt patterns + verification loops |
| Core | `0190_RAG_and_grounding.py` | Retrieve, then generate -- reduce hallucination |
| Core | `0200_tools_and_agents_lite.py` | Tools / function calling / light agents |
| Core | `0210_you_x_ai_delivery.py` | YOU x AI workflow -- ship a thin slice |

### Ch04: First training delivery -- see a model train

- Opener: `0220_Ch04_First_Delivery.py`
- Attempt: Run the training demo; change one knob; explain what happened out loud.

| Tag | File | Note |
|-----|------|------|
| Core | `0230_AI_fun.py` | Safe playground -- confidence first |
| Core | `0240_numpy_torch.py` | ndarray vs Tensor |
| Core | `0250_torch_training.py` | Lean training loop (+ notebook) |
| Lab | `0260_model.py` | Small end-to-end model toy |

### Ch05: Learning as binding

- Opener: `0270_Ch05_Learning_As_Binding.py`
- Attempt: Reclassify ImageNet, GPT pretrain, and RLHF in binding terms.

| Tag | File | Note |
|-----|------|------|
| Core | `0280_SL_UL_RL.py` | SL / UL / RL + binding climax |
| Core | `0290_regression_classification.py` | Two kinds of prediction questions |

### Ch06: Data and distribution

- Opener: `0300_Ch06_Data_And_Distribution.py`
- Attempt: Describe one real dataset's shift risk using 150_DL_future levels.

| Tag | File | Note |
|-----|------|------|
| Core | `0310_distribution.py` | Why distributions -- business gravity |
| Core | `0320_normalize_data.py` | Scale the data (not activation BN/LN) |
| Core | `0330_pipeline_feature_engineering.py` | Pipeline and features |

### Ch07: Evaluation and field use

- Opener: `0340_Ch07_Evaluation.py`
- Attempt: For a class-imbalanced task, argue why accuracy alone lies.

| Tag | File | Note |
|-----|------|------|
| Core | `0350_F1_Score.py` | Confusion matrix, precision, recall, F1 |
| Core | `0360_train_validate_test_deploy.py` | Train / validate / test / deploy |

### Ch08: Loss -- scoring the binding

- Opener: `0370_Ch08_Loss_Objectives.py`
- Attempt: Pick MSE vs MAE vs CE for one toy task and defend the choice.

| Tag | File | Note |
|-----|------|------|
| Core | `0380_loss_functions.py` | What a loss is |
| Core | `0390_Loss_Mean_Squared_Error.py` | MSE |
| Core | `0400_Loss_Mean_Absolute_Error.py` | MAE |
| Core | `0410_Loss_Huber_Loss.py` | Huber |
| Core | `0420_Loss_Binary_Cross_Entropy.py` | BCE |
| Core | `0430_Loss_Categorical_Cross_Entropy.py` | Categorical CE |
| Lab | `0440_Loss_Kullback_Leibler_Divergence.py` | KL |
| Core | `0450_maximum_likelihood.py` | MLE bridge to MSE / MAE |

### Ch09: Optimization -- how parameters move

- Opener: `0460_Ch09_Optimization.py`
- Attempt: Explain minibatch noise in one minute without equations.

| Tag | File | Note |
|-----|------|------|
| Core | `0470_derivative.py` | Derivatives / partials |
| Core | `0480_fastest_ascent.py` | Gradient direction |
| Core | `0490_rolling_ball.py` | Descent physics intuition |
| Core | `0500_stochastic_ascent.py` | Noisy steps still climb |
| Core | `0510_momentum.py` | Momentum under noise |
| Core | `0520_SGD_concept.py` | SGD concept |
| Core | `0530_SGD_training.py` | Full-batch vs minibatch |
| Lab | `0540_back_propagation.py` | Chain rule + autograd |
| Core | `0550_AdamW_and_LR_schedules.py` | Adam / AdamW + LR schedules |

### Ch10: Probability toolkit (just-in-time)

- Opener: `0560_Ch10_Probability_Toolkit.py`
- Attempt: State LLN vs CLT in one sentence each, with an example.

| Tag | File | Note |
|-----|------|------|
| Core | `0570_gaussian.py` | Gaussian basics |
| Core | `0580_law_of_large_numbers.py` | LLN |
| Core | `0590_central_limit_theorem.py` | CLT |
| Core | `0600_probability_vs_statistics.py` | Two directions |
| Core | `0610_Bayesian_theorem.py` | Bayes update |
| Core | `0620_exponential.py` | exp / log in ML |
| Enrichment | `0630_average_of_random.py` | Average of r.v.s |
| Enrichment | `0640_product_of_random.py` | Product of r.v.s |
| Enrichment | `0650_stress_test.py` | Misuse of averaging intuition |

### Ch11: What a net is -- function class and bias

- Opener: `0660_Ch11_Neural_Function_Class.py`
- Attempt: Why stacking linear layers without activations collapses.

| Tag | File | Note |
|-----|------|------|
| Core | `0670_inductive_bias.py` | Inductive bias |
| Core | `0680_neural_non_linearity.py` | Nonlinearity / manifolds |
| Core | `0690_dropout.py` | Dropout |
| Core | `0700_normalize_activations.py` | BatchNorm vs LayerNorm |
| TODO | `*(TODO)*` | CNN inductive bias |
| TODO | `*(TODO)*` | Initialization + residual networks |
| TODO | `*(TODO)*` | Overfitting / underfitting visual lecture |

### Ch12: Discriminative vs generative

- Opener: `0710_Ch12_Discriminative_Generative.py`
- Attempt: Name one task that needs P(x) and one that only needs P(y|x).

| Tag | File | Note |
|-----|------|------|
| Core | `0720_discriminative_generative.py` | Conceptual contrast |
| Lab | `0730_discriminative_toys.py` | Discriminative toys |
| Lab | `0740_generative_toys.py` | Generative toys |
| TODO | `*(TODO)*` | Diffusion intuition |

### Ch13: Representation -- tokens become geometry

- Opener: `0750_Ch13_Representation.py`
- Attempt: Explain why a word vector is useful without listing Word2Vec math.

| Tag | File | Note |
|-----|------|------|
| Core | `0760_Embedding.py` | Embedding tables |
| Core | `0770_word_vector.py` | Word2Vec intuition |
| Core | `0780_linear_transformation.py` | Linear maps as matrices |
| Core | `0790_dot_inner_scalar_product.py` | Dot product / similarity |
| Core | `0800_softmax.py` | Softmax |

### Ch14: Transformer -- deliver, then unpack

- Opener: `0810_Ch14_Transformer.py`
- Attempt: Run or read the off-the-shelf path; then explain one attention head.

| Tag | File | Note |
|-----|------|------|
| Core | `0820_off_the_shelf_transformer.py` | Call library blocks first |
| Core | `0830_transformer.py` | Self-attention intuition |
| Core | `0840_QKV_attention.py` | Q, K, V meanings |
| Core | `0850_Wq_Wk_Wv.py` | Learned projections |
| Core | `0860_positional_encoding.py` | Positions |
| Core | `0870_chain_of_adds.py` | Residual chain |
| Core | `0880_model_dimension.py` | Width vs sequence length |
| Lab | `0890_transformer_code.py` | Shapes from scratch |
| Core | `0900_Transformer_inductive_bias.py` | Transformer priors |
| Lab | `0910_machine_translation.py` | Tiny NMT |
| TODO | `*(TODO)*` | RNN / LSTM bridge -- problem attention solved |

### Ch15: LLM internals -- BERT / GPT mechanisms

- Opener: `0920_Ch15_LLMs.py`
- Attempt: Contrast masked pretraining vs next-token pretraining in binding words.

| Tag | File | Note |
|-----|------|------|
| Core | `0930_BERT.py` | Encoder pretraining + survey |
| Core | `0940_GPT.py` | Decoder LLMs, SFT path |
| TODO | `*(TODO)*` | RLHF / preference as external binding on top of pretrain |

### Ch16: Variants (enrichment)

- Opener: `0950_Ch16_Variants.py`
- Attempt: Name one reason global attention is expensive on images/video.

| Tag | File | Note |
|-----|------|------|
| Enrichment | `0960_video_transformer.py` | Video tokens / attention |
| Enrichment | `0970_swin_transformer.py` | Shifted windows |

### Parallel: Speaking clinic (parallel track)

- Opener: `0980_ChParallel_Speaking.py`
- Attempt: Answer one interview question twice: native clarity, then your voice.

| Tag | File | Note |
|-----|------|------|
| Core | `0990_Asian_interview.py` | Native vs Asian English bank |

## Minimal core path

- `0100_Ch01_Contract.py`
- `0110_lecture_strategy.py`
- `0120_intelligence.py`
- `0130_Ch02_Why_Deep_Learning.py`
- `0140_DL_history.py`
- `0150_DL_future.py`
- `0160_continuous_end_to_end.py`
- `0170_Ch03_LLM_Usage.py`
- `0180_prompting_and_verification.py`
- `0190_RAG_and_grounding.py`
- `0200_tools_and_agents_lite.py`
- `0210_you_x_ai_delivery.py`
- `0220_Ch04_First_Delivery.py`
- `0230_AI_fun.py`
- `0240_numpy_torch.py`
- `0250_torch_training.py`
- `0270_Ch05_Learning_As_Binding.py`
- `0280_SL_UL_RL.py`
- `0340_Ch07_Evaluation.py`
- `0360_train_validate_test_deploy.py`
- `0460_Ch09_Optimization.py`
- `0550_AdamW_and_LR_schedules.py`
- `0810_Ch14_Transformer.py`
- `0820_off_the_shelf_transformer.py`
- `0920_Ch15_LLMs.py`
- `0940_GPT.py`

## Notes

- File numbers are 4-digit zero-padded.
- Ch03 = LLM **usage**; Ch15 = LLM **internals**; Ch04 = first **training** delivery.

