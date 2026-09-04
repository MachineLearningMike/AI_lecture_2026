"""Patch _curriculum.py CHAPTERS + CORE_PATH for inserted Ch03 LLM Usage."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / "_curriculum.py"
text = path.read_text(encoding="utf-8")
start = text.index("# Chapters:")
end = text.index("CORE_PATH = [")
end2 = text.index("\n]", end) + 2
header = text[:start]
rest = text[end2:]

new_block = '''# Chapters: opener filename, title, blurb, attempt, rows of (tag, filename, note)
# Consecutive teaching numbers differ by 10. Ch03 = LLM usage (delivery-first).
CHAPTERS: list[dict] = [
    {
        "opener": "0100_Ch01_Contract.py",
        "code": "Ch01",
        "title": "Contract and gravity",
        "blurb": (
            "Why this course exists, and how we train:\\n"
            "begin delivery, advance toward understanding -- fast."
        ),
        "attempt": "Say Vision / Mission / Strategy / Attitude in your own words.",
        "rows": [
            ("Core", "0110_lecture_strategy.py", "Vision, mission, gravity, contract"),
            ("Core", "0120_intelligence.py", "Working definition of intelligence"),
        ],
    },
    {
        "opener": "0130_Ch02_Why_Deep_Learning.py",
        "code": "Ch02",
        "title": "Why deep learning",
        "blurb": (
            "Story before machinery: what DL already cracked,\\n"
            "where the frontier is, and why continuous end-to-end matters."
        ),
        "attempt": "Pick one capability x one generalization level; argue strong vs fragile.",
        "rows": [
            ("Core", "0140_DL_history.py", "Historical achievements and enabling ideas"),
            ("Core", "0150_DL_future.py", "Capability ladder x generalization axis"),
            ("Core", "0160_continuous_end_to_end.py", "Language-free continuous representations"),
        ],
    },
    {
        "opener": "0170_Ch03_LLM_Usage.py",
        "code": "Ch03",
        "title": "LLM usage -- deliver with models first",
        "blurb": (
            "Keep the gravity: squeeze LLMs for work before unpacking the spine.\\n"
            "Prompt, verify, ground with retrieval, call tools, ship a thin YOU x AI loop."
        ),
        "attempt": "Complete one real task with an LLM: prompt, verify, cite or refuse.",
        "rows": [
            ("Core", "0180_prompting_and_verification.py", "Prompt patterns + verification loops"),
            ("Core", "0190_RAG_and_grounding.py", "Retrieve, then generate -- reduce hallucination"),
            ("Core", "0200_tools_and_agents_lite.py", "Tools / function calling / light agents"),
            ("Core", "0210_you_x_ai_delivery.py", "YOU x AI workflow -- ship a thin slice"),
        ],
    },
    {
        "opener": "0240_Ch04_First_Delivery.py",
        "code": "Ch04",
        "title": "First training delivery -- see a model train",
        "blurb": (
            "After using LLMs, open the hood of a tiny trainer.\\n"
            "Play, touch tensors, watch a loss move."
        ),
        "attempt": "Run the training demo; change one knob; explain what happened out loud.",
        "rows": [
            ("Core", "0250_AI_fun.py", "Safe playground -- confidence first"),
            ("Core", "0260_numpy_torch.py", "ndarray vs Tensor"),
            ("Core", "0270_torch_training.py", "Lean training loop (+ notebook)"),
            ("Lab", "0280_model.py", "Small end-to-end model toy"),
        ],
    },
    {
        "opener": "0290_Ch05_Learning_As_Binding.py",
        "code": "Ch05",
        "title": "Learning as binding",
        "blurb": (
            "Learning creates relationships that were not given beforehand.\\n"
            "Externally-bound vs internally-bound -- cleaner than SL / UL slogans."
        ),
        "attempt": "Reclassify ImageNet, GPT pretrain, and RLHF in binding terms.",
        "rows": [
            ("Core", "0300_SL_UL_RL.py", "SL / UL / RL + binding climax"),
            ("Core", "0310_regression_classification.py", "Two kinds of prediction questions"),
        ],
    },
    {
        "opener": "0320_Ch06_Data_And_Distribution.py",
        "code": "Ch06",
        "title": "Data and distribution",
        "blurb": (
            "Start from the client's business and their data.\\n"
            "Scale features carefully; build a pipeline you can trust."
        ),
        "attempt": "Describe one real dataset's shift risk using 150_DL_future levels.",
        "rows": [
            ("Core", "0330_distribution.py", "Why distributions -- business gravity"),
            ("Core", "0340_normalize_data.py", "Scale the data (not activation BN/LN)"),
            ("Core", "0350_pipeline_feature_engineering.py", "Pipeline and features"),
        ],
    },
    {
        "opener": "0360_Ch07_Evaluation.py",
        "code": "Ch07",
        "title": "Evaluation and field use",
        "blurb": (
            "Delivery without judgment is theater.\\n"
            "Metrics, splits, deploy, and monitor."
        ),
        "attempt": "For a class-imbalanced task, argue why accuracy alone lies.",
        "rows": [
            ("Core", "0370_F1_Score.py", "Confusion matrix, precision, recall, F1"),
            ("Core", "0380_train_validate_test_deploy.py", "Train / validate / test / deploy"),
        ],
    },
    {
        "opener": "0390_Ch08_Loss_Objectives.py",
        "code": "Ch08",
        "title": "Loss -- scoring the binding",
        "blurb": (
            "The objective is how we score a proposed binding.\\n"
            "Teach after students have already watched training move."
        ),
        "attempt": "Pick MSE vs MAE vs CE for one toy task and defend the choice.",
        "rows": [
            ("Core", "0400_loss_functions.py", "What a loss is"),
            ("Core", "0410_Loss_Mean_Squared_Error.py", "MSE"),
            ("Core", "0420_Loss_Mean_Absolute_Error.py", "MAE"),
            ("Core", "0430_Loss_Huber_Loss.py", "Huber"),
            ("Core", "0440_Loss_Binary_Cross_Entropy.py", "BCE"),
            ("Core", "0450_Loss_Categorical_Cross_Entropy.py", "Categorical CE"),
            ("Lab", "0460_Loss_Kullback_Leibler_Divergence.py", "KL"),
            ("Core", "0470_maximum_likelihood.py", "MLE bridge to MSE / MAE"),
        ],
    },
    {
        "opener": "0480_Ch09_Optimization.py",
        "code": "Ch09",
        "title": "Optimization -- how parameters move",
        "blurb": (
            "Derivatives, gradients, noisy steps, momentum, backprop.\\n"
            "Math arrives because descent was already seen."
        ),
        "attempt": "Explain minibatch noise in one minute without equations.",
        "rows": [
            ("Core", "0490_derivative.py", "Derivatives / partials"),
            ("Core", "0500_fastest_ascent.py", "Gradient direction"),
            ("Core", "0510_rolling_ball.py", "Descent physics intuition"),
            ("Core", "0520_stochastic_ascent.py", "Noisy steps still climb"),
            ("Core", "0530_momentum.py", "Momentum under noise"),
            ("Core", "0540_SGD_concept.py", "SGD concept"),
            ("Core", "0550_SGD_training.py", "Full-batch vs minibatch"),
            ("Lab", "0560_back_propagation.py", "Chain rule + autograd"),
            ("Core", "0570_AdamW_and_LR_schedules.py", "Adam / AdamW + LR schedules"),
        ],
    },
    {
        "opener": "0580_Ch10_Probability_Toolkit.py",
        "code": "Ch10",
        "title": "Probability toolkit (just-in-time)",
        "blurb": (
            "Not a wall before delivery -- a toolkit pulled when losses\\n"
            "and evaluation need clearer language."
        ),
        "attempt": "State LLN vs CLT in one sentence each, with an example.",
        "rows": [
            ("Core", "0590_gaussian.py", "Gaussian basics"),
            ("Core", "0600_law_of_large_numbers.py", "LLN"),
            ("Core", "0610_central_limit_theorem.py", "CLT"),
            ("Core", "0620_probability_vs_statistics.py", "Two directions"),
            ("Core", "0630_Bayesian_theorem.py", "Bayes update"),
            ("Core", "0640_exponential.py", "exp / log in ML"),
            ("Enrichment", "0650_average_of_random.py", "Average of r.v.s"),
            ("Enrichment", "0660_product_of_random.py", "Product of r.v.s"),
            ("Enrichment", "0670_stress_test.py", "Misuse of averaging intuition"),
        ],
    },
    {
        "opener": "0680_Ch11_Neural_Function_Class.py",
        "code": "Ch11",
        "title": "What a net is -- function class and bias",
        "blurb": (
            "Architecture is prior.\\n"
            "Nonlinearity, regularization, activation normalization."
        ),
        "attempt": "Why stacking linear layers without activations collapses.",
        "rows": [
            ("Core", "0690_inductive_bias.py", "Inductive bias"),
            ("Core", "0700_neural_non_linearity.py", "Nonlinearity / manifolds"),
            ("Core", "0710_dropout.py", "Dropout"),
            ("Core", "0720_normalize_activations.py", "BatchNorm vs LayerNorm"),
            ("TODO", "", "CNN inductive bias"),
            ("TODO", "", "Initialization + residual networks"),
            ("TODO", "", "Overfitting / underfitting visual lecture"),
        ],
    },
    {
        "opener": "0730_Ch12_Discriminative_Generative.py",
        "code": "Ch12",
        "title": "Discriminative vs generative",
        "blurb": (
            "Two readings of a distribution:\\n"
            "draw a boundary, or model how the world looks."
        ),
        "attempt": "Name one task that needs P(x) and one that only needs P(y|x).",
        "rows": [
            ("Core", "0740_discriminative_generative.py", "Conceptual contrast"),
            ("Lab", "0750_discriminative_toys.py", "Discriminative toys"),
            ("Lab", "0760_generative_toys.py", "Generative toys"),
            ("TODO", "", "Diffusion intuition"),
        ],
    },
    {
        "opener": "0770_Ch13_Representation.py",
        "code": "Ch13",
        "title": "Representation -- tokens become geometry",
        "blurb": (
            "Embeddings, similarity, and softmax.\\n"
            "Linear algebra as the language of attention -- not abstract prelude."
        ),
        "attempt": "Explain why a word vector is useful without listing Word2Vec math.",
        "rows": [
            ("Core", "0780_Embedding.py", "Embedding tables"),
            ("Core", "0790_word_vector.py", "Word2Vec intuition"),
            ("Core", "0800_linear_transformation.py", "Linear maps as matrices"),
            ("Core", "0810_dot_inner_scalar_product.py", "Dot product / similarity"),
            ("Core", "0820_softmax.py", "Softmax"),
        ],
    },
    {
        "opener": "0830_Ch14_Transformer.py",
        "code": "Ch14",
        "title": "Transformer -- deliver, then unpack",
        "blurb": (
            "Call a Transformer first.\\n"
            "Then name attention, QKV, positions, residuals, and inductive bias."
        ),
        "attempt": "Run or read the off-the-shelf path; then explain one attention head.",
        "rows": [
            ("Core", "0840_off_the_shelf_transformer.py", "Call library blocks first"),
            ("Core", "0850_transformer.py", "Self-attention intuition"),
            ("Core", "0860_QKV_attention.py", "Q, K, V meanings"),
            ("Core", "0870_Wq_Wk_Wv.py", "Learned projections"),
            ("Core", "0880_positional_encoding.py", "Positions"),
            ("Core", "0890_chain_of_adds.py", "Residual chain"),
            ("Core", "0900_model_dimension.py", "Width vs sequence length"),
            ("Lab", "0910_transformer_code.py", "Shapes from scratch"),
            ("Core", "0920_Transformer_inductive_bias.py", "Transformer priors"),
            ("Lab", "0930_machine_translation.py", "Tiny NMT"),
            ("TODO", "", "RNN / LSTM bridge -- problem attention solved"),
        ],
    },
    {
        "opener": "0940_Ch15_LLMs.py",
        "code": "Ch15",
        "title": "LLM internals -- BERT / GPT mechanisms",
        "blurb": (
            "Architecture and pretraining -- how LLMs are built.\\n"
            "Usage/tooling already lived in Ch03; here we unpack the spine products."
        ),
        "attempt": "Contrast masked pretraining vs next-token pretraining in binding words.",
        "rows": [
            ("Core", "0950_BERT.py", "Encoder pretraining + survey"),
            ("Core", "0960_GPT.py", "Decoder LLMs, SFT path"),
            ("TODO", "", "RLHF / preference as external binding on top of pretrain"),
        ],
    },
    {
        "opener": "0970_Ch16_Variants.py",
        "code": "Ch16",
        "title": "Variants (enrichment)",
        "blurb": (
            "After the spine: video and hierarchical vision Transformers.\\n"
            "Optional depth -- not required for the core path."
        ),
        "attempt": "Name one reason global attention is expensive on images/video.",
        "rows": [
            ("Enrichment", "0980_video_transformer.py", "Video tokens / attention"),
            ("Enrichment", "0990_swin_transformer.py", "Shifted windows"),
        ],
    },
    {
        "opener": "1000_ChParallel_Speaking.py",
        "code": "Parallel",
        "title": "Speaking clinic (parallel track)",
        "blurb": (
            "Attitude and career resilience -- talk about AI clearly.\\n"
            "Not a deep-learning chapter; run beside the spine."
        ),
        "attempt": "Answer one interview question twice: native clarity, then your voice.",
        "rows": [
            ("Core", "1010_Asian_interview.py", "Native vs Asian English bank"),
        ],
    },
]

CORE_PATH = [
    "0100_Ch01_Contract.py",
    "0110_lecture_strategy.py",
    "0120_intelligence.py",
    "0130_Ch02_Why_Deep_Learning.py",
    "0140_DL_history.py",
    "0150_DL_future.py",
    "0160_continuous_end_to_end.py",
    "0170_Ch03_LLM_Usage.py",
    "0180_prompting_and_verification.py",
    "0190_RAG_and_grounding.py",
    "0200_tools_and_agents_lite.py",
    "0210_you_x_ai_delivery.py",
    "0240_Ch04_First_Delivery.py",
    "0250_AI_fun.py",
    "0260_numpy_torch.py",
    "0270_torch_training.py",
    "0290_Ch05_Learning_As_Binding.py",
    "0300_SL_UL_RL.py",
    "0360_Ch07_Evaluation.py",
    "0380_train_validate_test_deploy.py",
    "0480_Ch09_Optimization.py",
    "0570_AdamW_and_LR_schedules.py",
    "0830_Ch14_Transformer.py",
    "0840_off_the_shelf_transformer.py",
    "0940_Ch15_LLMs.py",
    "0960_GPT.py",
]
'''

# Fix escaped newlines in the triple-quoted Python source: we used \\n in strings
# so the written file contains literal \n escape sequences for Python source. Good.

path.write_text(header + new_block + rest, encoding="utf-8")
print("patched", path)
