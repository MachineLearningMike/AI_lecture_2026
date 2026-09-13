# AI Lecture 2026

Slide-deck course material. Each lecture is a standalone Python file.

## How to run

Activate the lecture environment, then either:

```bash
python _dashboard.py          # pick any lecture from the launcher
python 7_02_LLM_opportunities.py   # run one module directly
```

Interactive slides (matplotlib windows, ←/→ navigation, **sliders**) need a GUI backend. Do not use `MPLBACKEND=Agg` for live teaching.

Optional GIF export for animations:

```bash
set LECTURE_EXPORT_GIF=1
python <module>.py
```

## Independence rules

- Every `N_MM_*.py` lecture runs on its own.
- Lectures do **not** import each other.
- The only shared project module lectures should use is `_config.py` (fonts, footer, slide navigation, widget retention).
- `_dashboard.py` is the optional launcher, not a lecture dependency.
- No cross-lecture “see module X / Ch0Y” coupling in slide text.

## Sliders

About **34** lectures use `matplotlib.widgets.Slider` for interactive demos (Bayes priors, thresholds, RL knobs, loss curves, and similar).

- Sliders live **inside** that lecture file; they do not call other lectures.
- `_config.run_slide_deck` installs widget retention and hardens bottom-axis interaction so sliders are not garbage-collected and do not deadlock with the ←/→ nav buttons.
- Smoke tests under `Agg` only prove import/build; drag the sliders in a normal GUI window to verify interactivity.

## Material counts

| Item | Count |
|---|---:|
| Slide files (`N_MM_*.py`) | 115 |
| Chapter openers (`N_00_*.py`) | 10 |
| Subject files | 105 |
| Companion notebooks (`.ipynb`) | 9 |
| Modules with sliders | 34 |

Runtime helpers: `_config.py`, `_dashboard.py`.

## By chapter

| Ch | Title | Files | Subjects |
|---:|---|---:|---:|
| 0 | Preface | 4 | 3 |
| 1 | Data | 18 | 17 |
| 2 | Learning | 14 | 13 |
| 3 | Training | 34 | 33 |
| 4 | Networks | 3 | 2 |
| 5 | Transformer | 15 | 14 |
| 6 | Reinforcement | 6 | 5 |
| 7 | LLM and VLM | 10 | 9 |
| 8 | Representation | 8 | 7 |
| 9 | Agents | 3 | 2 |

## Chapters and slide files

### 0 — Preface

- `0_00_Preface.py`
- `0_02_lecture_strategy.py`
- `0_04_intelligence.py`
- `0_06_Asian_interview.py`

### 1 — Data

- `1_00_Data.py`
- `1_02_distribution.py`
- `1_04_probability_statistics_terms.py`
- `1_06_probability_density.py`
- `1_08_gaussian.py`
- `1_10_more_distributions.py`
- `1_12_law_of_large_numbers.py`
- `1_14_central_limit_theorem.py`
- `1_16_probability_vs_statistics.py`
- `1_18_Bayes_theorem.py`
- `1_20_naive_bayesian_classifier.py`
- `1_22_sum_of_randomity.py`
- `1_24_average_of_randomity.py`
- `1_26_product_of_randomity.py`
- `1_28_stress_test.py`
- `1_30_normalize_data.py`
- `1_32_pipeline_feature_engineering.py`
- `1_34_F1_Score.py`

### 2 — Learning

- `2_00_Learning.py`
- `2_02_AI_fun.py`
- `2_04_learning.py`
- `2_06_maximum_likelihood_estimation.py`
- `2_08_other_learning_methods.py`
- `2_10_SL_UL_RL.py`
- `2_12_continuous_end_to_end.py`
- `2_14_DL_history.py`
- `2_16_DL_future.py`
- `2_18_regression_classification.py`
- `2_20_discriminative_generative.py`
- `2_22_discriminative_toys.py`
- `2_24_generative_toys.py`
- `2_26_Monte_Carlo.py`

### 3 — Training

- `3_00_Training.py`
- `3_02_train_validate_test.py`
- `3_04_numpy_torch.py`
- `3_06_torch_training.py`
- `3_08_train_simple_complec.py`
- `3_10_train_regression_MLP.py`
- `3_12_train_multi_classifier_MLP.py`
- `3_14_train_multi_classifier_RNN.py`
- `3_16_train_next_predictor_RNN.py`
- `3_18_loss_functions.py`
- `3_20_Loss_Mean_Squared_Error.py`
- `3_22_Loss_Mean_Absolute_Error.py`
- `3_24_Loss_Huber_Loss.py`
- `3_26_Loss_Binary_Cross_Entropy.py`
- `3_28_Loss_Categorical_Cross_Entropy.py`
- `3_30_Loss_Kullback_Leibler_Divergence.py`
- `3_32_derivative.py`
- `3_34_fastest_ascent.py`
- `3_36_rolling_ball.py`
- `3_38_stochastic_ascent.py`
- `3_40_SGD_concept.py`
- `3_42_SGD_training.py`
- `3_44_back_propagation.py`
- `3_46_AdamW_and_LR_schedules.py`
- `3_48_exponential.py`
- `3_50_inductive_bias.py`
- `3_52_neural_non_linearity.py`
- `3_54_dropout.py`
- `3_56_normalize_activations.py`
- `3_58_Embedding.py`
- `3_60_softmax.py`
- `3_62_data_pipeline.py`
- `3_64_momentum.py`
- `3_66_YOLO_training.py`

### 4 — Networks

- `4_00_Networks.py`
- `4_02_major_CNNs.py`
- `4_04_major_RNNs.py`

### 5 — Transformer

- `5_00_Transformer.py`
- `5_02_linear_transformation.py`
- `5_04_dot_inner_scalar_product.py`
- `5_06_off_the_shelf_transformer.py`
- `5_08_transformer.py`
- `5_10_QKV_attention.py`
- `5_12_Wq_Wk_Wv.py`
- `5_14_positional_encoding.py`
- `5_16_chain_of_adds.py`
- `5_18_model_dimension.py`
- `5_20_transformer_code.py`
- `5_22_Transformer_inductive_bias.py`
- `5_24_machine_translation.py`
- `5_26_video_transformer.py`
- `5_28_swin_transformer.py`

### 6 — Reinforcement

- `6_00_Reinforcement.py`
- `6_02_Markov_Decision_Process.py`
- `6_04_Bellman_equality.py`
- `6_06_policy_value_iteration.py`
- `6_08_policy_gradient_methods.py`
- `6_10_major_RL_principles.py`

### 7 — LLM and VLM

- `7_00_LLM_and_VLM.py`
- `7_02_LLM_opportunities.py`
- `7_04_train_ChatGPT.py`
- `7_06_exploit_stage1_talk.py`
- `7_08_exploit_stage2_data.py`
- `7_10_exploit_stage3_compute.py`
- `7_12_exploit_stage4_systems.py`
- `7_14_exploit_eval_production.py`
- `7_16_exploit_deep_progression.py`
- `7_18_Vision_Language_Model.py`

### 8 — Representation

- `8_00_Representation.py`
- `8_02_word_vector.py`
- `8_04_word_to_vector.py`
- `8_06_BERT.py`
- `8_08_GPT.py`
- `8_10_VAE.py`
- `8_12_GAN.py`
- `8_14_Diffusion.py`

### 9 — Agents

- `9_00_Agents.py`
- `9_02_agentic_AI.py`
- `9_04_agentic_AI_2.py`

## Companion notebooks

- `3_06_torch_training.ipynb`
- `3_10_train_regression_MLP.ipynb`
- `3_12_train_multi_classifier_MLP.ipynb`
- `3_14_train_multi_classifier_RNN.ipynb`
- `3_16_train_next_predictor_RNN.ipynb`
- `3_40_SGD_code.ipynb`
- `5_20_transformer_code_alt.ipynb`
- `5_20_transformer_code_scratch.ipynb`
- `5_24_machine_translation.ipynb`
