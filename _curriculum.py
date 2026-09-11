"""
_curriculum.py

Single source of truth for renames and the teaching spine (CORE_PATH).
Chapter-opener preface scripts (*_Ch*.py) were removed; teach by CORE_PATH order.
GAP10 applied: consecutive teaching files differ by 10 in their prefix.
"""

from __future__ import annotations

# old_name -> new_name (files that currently exist and will be renamed)
RENAMES: dict[str, str] = {
    # Ch01
    "010_lecture_strategy.py": "0110_lecture_strategy.py",
    "030_intelligence.py": "0120_intelligence.py",
    # Ch02
    "012_DL_history.py": "0140_DL_history.py",
    "013_DL_future.py": "0150_DL_future.py",
    "020_continuous_end_to_end.py": "0160_continuous_end_to_end.py",
    # Ch03
    "041_AI_fun.py": "0250_AI_fun.py",
    "043_numpy_torch.py": "0260_numpy_torch.py",
    "016_torch_training.py": "0270_torch_training.py",
    "080_model.py": "0280_model.py",
    "015_torch_training.ipynb": "0270_torch_training.ipynb",
    "016_torch_training.gif": "0270_torch_training.gif",
    # Ch04
    "014_SL_UL_RL.py": "0300_SL_UL_RL.py",
    "035_regression_classification.py": "0310_regression_classification.py",
    # Ch05
    "040_distribution.py": "0330_distribution.py",
    "042_normalize.py": "0340_normalize_data.py",
    "231_pipeline_feature_engineering.py": "0350_pipeline_feature_engineering.py",
    # Ch06
    "122_F1_Score.py": "0370_F1_Score.py",
    "125_train_validate_test_deploy.py": "0380_train_validate_test_deploy.py",
    # Ch07
    "185_loss_functions.py": "0400_loss_functions.py",
    "191_Loss_Mean_Squared_Error.py": "0410_Loss_Mean_Squared_Error.py",
    "192_Loss_Mean_Absolute_Error.py": "0420_Loss_Mean_Absolute_Error.py",
    "193_Loss_Huber_Loss.py": "0430_Loss_Huber_Loss.py",
    "194_Loss_Binary_Cross_Entropy.py": "0440_Loss_Binary_Cross_Entropy.py",
    "195_Loss_Categorical_Cross_Entropy.py": "0450_Loss_Categorical_Cross_Entropy.py",
    "196_Loss_Kullback_Leibler_Divergence.py": "0460_Loss_Kullback_Leibler_Divergence.py",
    "120_maximum_likelihood.py": "0470_maximum_likelihood.py",
    # Ch08
    "140_derivative.py": "0490_derivative.py",
    "150_fastest_ascent.py": "0500_fastest_ascent.py",
    "170_rolling_ball.py": "0510_rolling_ball.py",
    "160_stochastic_ascent.py": "0520_stochastic_ascent.py",
    "180_momentum.py": "0530_momentum.py",
    "197_SGD_concept.py": "0540_SGD_concept.py",
    "210_SGD_training.py": "0550_SGD_training.py",
    "198_back_propagation.py": "0560_back_propagation.py",
    "143_SGD_code.ipynb": "0540_SGD_code.ipynb",
    # Ch09
    "060_gaussian.py": "0590_gaussian.py",
    "085_law_of_large_numbers.py": "0600_law_of_large_numbers.py",
    "090_central_limit_theorem.py": "0610_central_limit_theorem.py",
    "115_probability_vs_statistics.py": "0620_probability_vs_statistics.py",
    "095_Bayesian_theorem.py": "0630_Bayesian_theorem.py",
    "130_exponential.py": "0640_exponential.py",
    "100_average_of_random.py": "0650_average_of_random.py",
    "110_product_of_random.py": "0660_product_of_random.py",
    "116_stress_test.py": "0670_stress_test.py",
    # Ch10
    "220_inductive_bias.py": "0690_inductive_bias.py",
    "250_neural_non_linearity.py": "0700_neural_non_linearity.py",
    "240_dropout.py": "0710_dropout.py",
    "230_normalization.py": "0720_normalize_activations.py",
    # Ch11
    "251_discriminative_generative.py": "0740_discriminative_generative.py",
    "141_discriminative.py": "0750_discriminative_toys.py",
    "142_generative.py": "0760_generative_toys.py",
    # Ch12
    "210_Embedding.py": "0780_Embedding.py",
    "199_word_vector.py": "0790_word_vector.py",
    "300_linear_transformation.py": "0800_linear_transformation.py",
    "310_dot_inner_scalar_product.py": "0810_dot_inner_scalar_product.py",
    "320_softmax.py": "0820_softmax.py",
    # Ch13
    "290_off_the_shelf_transformer.py": "0840_off_the_shelf_transformer.py",
    "260_transformer.py": "0850_transformer.py",
    "330_QKV_attention.py": "0860_QKV_attention.py",
    "340_Wq_Wk_Wv.py": "0870_Wq_Wk_Wv.py",
    "350_positional_encoding.py": "0880_positional_encoding.py",
    "360_chain_of_adds.py": "0890_chain_of_adds.py",
    "370_model_dimension.py": "0900_model_dimension.py",
    "270_transformer_code.py": "0910_transformer_code.py",
    "380_Transformer_inductive_bias.py": "0920_Transformer_inductive_bias.py",
    "280_machine_translation.py": "0930_machine_translation.py",
    "210_machine_translation.ipynb": "0930_machine_translation.ipynb",
    "151_transformer_code.ipynb": "0910_transformer_code_scratch.ipynb",
    "209_transformer_code.ipynb": "0910_transformer_code_alt.ipynb",
    # Ch14
    "410_BERT.py": "0950_BERT.py",
    "420_GPT.py": "0960_GPT.py",
    # Ch15
    "390_video_transformer.py": "0980_video_transformer.py",
    "400_swin_transformer.py": "0990_swin_transformer.py",
    # Parallel
    "011_Asian_interview.py": "1010_Asian_interview.py",
}

# Teaching spine (no chapter-opener prefaces).
# Filenames include the current track prefix used on disk (e.g. 7_0232_...).
CORE_PATH = [
    "0_010_lecture_strategy.py",
    "0_020_intelligence.py",
    "1_020_gaussian.py",
    "1_025_more_distributions.py",
    "2_060_DL_history.py",
    "2_070_DL_future.py",
    "2_050_continuous_end_to_end.py",
    "7_010_Opportunity.py",
    "7_020_train_ChatGPT.py",
    "7_030_exploit_stage1_talk.py",
    "7_040_exploit_stage2_data.py",
    "7_050_exploit_stage3_compute.py",
    "7_060_exploit_stage4_systems.py",
    "7_070_exploit_eval_production.py",
    "7_080_exploit_deep_progression.py",
    "7_090_Vision_Language_Model.py",
    "9_010_agentic_AI.py",
    "9_020_agentic_AI_example.py",
    "2_010_AI_fun.py",
    "3_020_numpy_torch.py",
    "3_030_torch_training.py",
    "3_265_data_pipeline.py",
    "3_270_YOLO_training.py",
    "4_010_linear_transformation.py",
    "4_020_major_CNNs.py",
    "2_040_SL_UL_RL.py",
    "3_010_train_validate_test_deploy.py",
    "3_190_AdamW_and_LR_schedules.py",
    "5_020_off_the_shelf_transformer.py",
    "8_040_GPT.py",
]




