"""
011_Asian_interview.py

AI/ML interview speaking bank -- one question per slide, TWO answers:

  Left  = Native English (clean, natural)
  Right = Asian English (noisier grammar; intentional L1-like errors)

«...» on the Asian side marks deliberate twists; shown as [*k] with a red legend.

Do NOT memorize word-for-word. Memorize the skeleton, then speak.
"""

from __future__ import annotations

import re
import textwrap

import matplotlib.pyplot as plt
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_ANNOT,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    add_step_footer,
)
apply_rcparams(plt)
from matplotlib.patches import FancyBboxPatch


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return (w / dpi * fraction, h / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


def new_figure(*args, **kwargs):
    kwargs.setdefault('figsize', screen_figsize())
    kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, 'manager', None)
    if manager is not None:
        try:
            manager.window.state('zoomed')
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                try:
                    manager.frame.Maximize(True)
                except Exception:
                    pass
    add_step_footer(fig)
    plt.show()


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')


def round_box(ax, x, y, w, h, facecolor, edgecolor='0.2', lw=2.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'
C_TWIST = '#922b21'
C_NATIVE = '#eafaf1'
C_ASIAN = '#fdebd0'

MARK_RE = re.compile(r'«([^»]+)»')


def extract_twists(answer: str) -> tuple[str, list[str]]:
    twists = MARK_RE.findall(answer)
    out = answer
    for i, span in enumerate(twists, start=1):
        out = out.replace(f'«{span}»', f'{span}[*{i}]', 1)
    return out, twists


def wrap_block(text: str, width: int = 42) -> str:
    paragraphs = text.split('\n')
    wrapped = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            wrapped.append('')
        else:
            wrapped.extend(textwrap.wrap(p, width=width) or [''])
    return '\n'.join(wrapped)


# BANK item: (section, question, native_answer, asian_answer_with_«twists»)
BANK: list[tuple[str, str, str, str]] = [
    (
        'A. Mindset',
        'Tell me about yourself.',
        'I am an AI engineer, but my background is broader than AI.\n'
        'I spent many years in software and systems, then moved deeper into machine learning.\n\n'
        'What interests me most is not using a particular model.\n'
        'I like finding what is hidden behind the data, then designing the model around that.',
        'I am «AI engineer», but background is broader than AI.\n'
        'I «spend» many years in software and systems, then more deep into ML.\n\n'
        'What interest me most is «not using particular model».\n'
        'I like find what hidden behind data, then «design model around that».',
    ),
    (
        'A. Mindset',
        'What is your approach to an unfamiliar ML problem?',
        'First I try not to touch the model. I want to understand the data-generating process.\n\n'
        'What is observed? What is missing? What is actually the target?\n'
        'Many ML problems become different problems after you understand this.',
        'First I try «not touch the model». I want understand «data-generating process».\n\n'
        'What is observed? What missing? What is actually the target?\n'
        'Many ML problem become different problem after you understand «this».',
    ),
    (
        'A. Mindset',
        'How do you decide which model to use?',
        'I do not start from "Transformer or CNN?" I start from the structure of the problem.\n\n'
        'Spatial? Temporal? Graph? Maybe attention makes sense. Maybe it does not.\n'
        'The architecture should follow the structure, not fashion.',
        'I don\'t start from "Transformer or CNN?"\n'
        'I start from «structure of the problem».\n\n'
        'Spatial? Temporal? Graph? Maybe attention make sense. Maybe not.\n'
        'Architecture should follow the structure, «not fashion».',
    ),
    (
        'A. Mindset',
        'What makes a good ML engineer?',
        'Partly knowing when not to use ML.\n\n'
        'And after deciding to use ML: knowing what information the model should see,\n'
        'and what information we should not destroy before giving it to the model.',
        'Partly knowing «when not to use ML».\n\n'
        'And after decide use ML: what information model should see,\n'
        'and what information we should «not destroy» before give to model.',
    ),
    (
        'A. Mindset',
        'What is your biggest strength?',
        'I tend to go underneath the conventional formulation.\n\n'
        'If everyone says this is a classification problem, I may ask whether classification\n'
        'is really the problem. Maybe there is a continuous state behind the classes.',
        'I tend go underneath conventional formulation.\n\n'
        'If everyone say this is classification problem, I may ask:\n'
        'classification is really the problem?\n'
        'Maybe there is «continuous state behind the classes».',
    ),
    (
        'A. Mindset',
        'What is your weakness?',
        'I sometimes go too deep into the underlying mechanism.\n'
        'That is useful for difficult problems, but it can be inefficient.\n\n'
        'I have to remind myself: first make something work, then investigate why it works.',
        'Sometimes I go too deep into underlying mechanism.\n'
        'Useful for difficult problem, but can be inefficient.\n\n'
        'I remind myself: «first make something work», then investigate why «it works».',
    ),
    (
        'A. Mindset',
        'How do you handle disagreement with another engineer?',
        'I do not care much whose idea it is.\n'
        'I want an experiment that can kill one of the ideas.\n\n'
        'If my idea loses, good. We learned something.',
        'I don\'t care much whose idea it is.\n'
        'I want experiment that can «kill one of the ideas».\n\n'
        'If my idea lose, good. We learned «something».',
    ),
    (
        'A. Mindset',
        'How do you deal with uncertainty?',
        'I try not to hide it.\n\n'
        'If the model gives 0.8 but the input is outside the learned distribution,\n'
        '0.8 may be a very dangerous number.\n\n'
        'I want the model to know when its prediction is trustworthy.',
        'I try not hide it.\n\n'
        'If model give 0.8 but input is outside learned distribution,\n'
        '0.8 maybe is «very dangerous number».\n\n'
        'I want model know when prediction is «trustworthy».',
    ),
    (
        'B. Data / representation',
        'What is more important: data or model?',
        'Usually data. But I would make one correction: representation of data.\n\n'
        'A bad representation can make good data look useless.\n'
        'A very large model cannot always recover information that we destroyed\n'
        'before giving it to the model.',
        'Usually data. But one correction: «representation of data».\n\n'
        'Bad representation can make good data look useless.\n'
        'Very large model cannot always recover informations\n'
        'that we destroyed before give to «the model».',
    ),
    (
        'B. Data / representation',
        'What is feature engineering to you?',
        'It is human knowledge injected into the model.\n\n'
        'Sometimes very useful. But there is a cost: we are also injecting our assumptions.\n'
        'If we have enough data, I prefer the model to discover the representation.',
        'It is «human knowledge injected into the model».\n\n'
        'Sometimes very useful. But there is cost. We also inject our assumptions.\n'
        'If we have enough data, I prefer model «discover representation».',
    ),
    (
        'B. Data / representation',
        'Would you remove all feature engineering?',
        'No. That would be another dogma.\n\n'
        'If I know something physically must be true, I do not see a reason to make\n'
        'the network rediscover it from ten billion examples.\n\n'
        'The question is where human knowledge helps, and where it becomes a limitation.',
        'No. That would be another dogma.\n\n'
        'If I know something physically must true, I don\'t see reason make network\n'
        'rediscover it from ten billion example.\n\n'
        'Question is where human knowledge help, and where it become «limitation».',
    ),
    (
        'B. Data / representation',
        'What is a good representation?',
        'Something that preserves the distinctions important for the task,\n'
        'while removing irrelevant variation.\n\n'
        'But "irrelevant" depends on the task. That is why representation learning is difficult.',
        'Something preserve distinctions important for the task,\n'
        'while remove irrelevant variation.\n\n'
        'But "irrelevant" depend on the task.\n'
        '«That is why representation learning is difficult.»',
    ),
    (
        'B. Data / representation',
        'What do you mean by latent representation?',
        'Something we do not directly observe, but which explains many observed things.\n\n'
        'There must be something latent that represents the situation with enough fidelity --\n'
        'something we can bend into a useful indicator.\n\n'
        'I do not need every dimension to have a human meaning.',
        'Something we don\'t directly observe, but explain many observed things.\n\n'
        'There must be something latent, representing situation with enough fidelity.\n'
        '«Something we can bend into useful indicator.»\n\n'
        'I don\'t need every dimension have «human meaning».',
    ),
    (
        'B. Data / representation',
        'Do you think neural representations are interpretable?',
        'Not necessarily in the human sense.\n\n'
        'A representation can be meaningful for prediction without corresponding to a word\n'
        'like "temperature" or "dog."\n\n'
        'We sometimes expect the network to think in our vocabulary. Why?',
        'Not necessarily in human sense.\n\n'
        'Representation can be meaningful for prediction without correspond to word\n'
        'like "temperature" or "dog."\n\n'
        'We sometimes expect network think in our vocabulary. «Why?»',
    ),
    (
        'B. Data / representation',
        'What is information loss in ML?',
        'It happens everywhere: normalization, discretization, pooling, quantization,\n'
        'feature selection.\n\n'
        'Some loss is necessary. But I want to know what I am throwing away.\n'
        'Especially early in the pipeline, I am careful about irreversible operations.',
        'It happen everywhere: normalization, discretization, pooling, quantization,\n'
        'feature selection.\n\n'
        'Some loss is necessary. But I want know «what I am throwing away».\n'
        'Especially early in pipeline, I careful about irreversible «operations».',
    ),
    (
        'C. Classical ML',
        'Explain overfitting.',
        'The simple explanation is that the model fits the training data too specifically.\n\n'
        'But "big model equals overfitting" is not correct. A large model can learn a very\n'
        'useful structure. The real question is what exactly it memorized.',
        'Simple explanation: model fit training data too specifically.\n\n'
        'But "big model equal overfitting" -- I don\'t think this is correct.\n'
        'Large model can learn very useful structure.\n'
        'Real question is: «what exactly did it memorize?»',
    ),
    (
        'C. Classical ML',
        'How do you prevent overfitting?',
        'More data is usually my first answer.\n'
        'Then regularization, augmentation, architecture, early stopping.\n\n'
        'But I prefer to understand why it is overfitting before adding regularization everywhere.',
        'More data usually is my first answer.\n'
        'Then regularization, augmentation, architecture, early stopping.\n\n'
        'But I prefer understand why it is overfitting\n'
        '«before adding regularization everywhere».',
    ),
    (
        'C. Classical ML',
        'Why does regularization work?',
        'It restricts the solutions the model is allowed to prefer.\n\n'
        'In some sense we are saying: among models that explain the data,\n'
        'prefer some simpler or smoother behavior.',
        'It restrict the solutions model is allowed to prefer.\n\n'
        'In some sense we are saying: among models that explain the data,\n'
        'prefer some «simpler or smoother behavior».',
    ),
    (
        'C. Classical ML',
        'What is the bias-variance tradeoff?',
        'Bias is roughly: your model cannot represent the true relationship well.\n'
        'Variance is: it reacts too much to the particular training sample.\n\n'
        'So one is too rigid, and the other is too sensitive.',
        'Bias is roughly: your model cannot represent true relationship well.\n'
        'Variance is: it react too much to particular training sample.\n\n'
        'So one is too rigid, the other is «too sensitive».',
    ),
    (
        'C. Classical ML',
        'How do you handle imbalanced data?',
        'First I ask whether the imbalance is actually a problem. Sometimes it is the real world.\n\n'
        'If the rare class is what matters, accuracy is almost useless.\n'
        'I look at the cost of different errors and choose the metric from there.',
        'First I ask whether imbalance is actually problem.\n'
        'Sometimes it is «the real world».\n\n'
        'If rare class is what matter, accuracy almost useless.\n'
        'I look at cost of different errors, then choose metric from «there».',
    ),
    (
        'D. Deep learning',
        'Why do neural networks work so well?',
        'I do not think we have one satisfying answer.\n'
        'Universal approximation is not enough to explain practical deep learning.\n\n'
        'My intuition is that depth lets us build complicated representations through many\n'
        'relatively simple transformations. Composition matters.',
        'I don\'t think we have one satisfying answer.\n'
        'Universal approximation is not enough explain practical deep learning.\n\n'
        'My intuition: depth give way to build complicated representations\n'
        'through many relatively simple transformation. «Composition matters.»',
    ),
    (
        'D. Deep learning',
        'Why depth instead of one huge layer?',
        'Because composition matters.\n\n'
        'Some functions are much easier to represent as a sequence of transformations\n'
        'than as one flat transformation.\n'
        'The network can progressively change the representation.',
        'Because composition matters.\n\n'
        'Some function much easier as sequence of transformations\n'
        'than as one flat transformation.\n'
        'Network can «progressively change the representation».',
    ),
    (
        'D. Deep learning',
        'Why do we need nonlinear activation functions?',
        'Without nonlinearity, many layers collapse into one linear map.\n\n'
        'So depth alone does not give us much.\n'
        'The nonlinearities are what make composition interesting.',
        'Without nonlinearity, many layers collapse into one linear map.\n\n'
        'So depth alone don\'t give us much.\n'
        'Nonlinearities are what make composition «interesting».',
    ),
    (
        'D. Deep learning',
        'Why ReLU?',
        'It is simple, cheap, and gives useful gradient behavior compared with saturating\n'
        'activations.\n\n'
        'But I would not say ReLU is sacred. GELU, SiLU, and other choices can work very well.',
        'It is simple, cheap, and give useful gradient behavior\n'
        'compare with saturating activations.\n\n'
        'But I wouldn\'t say ReLU is sacred.\n'
        'GELU, SiLU, other choice can work very well. «Not dogma.»',
    ),
    (
        'D. Deep learning',
        'Why normalization?',
        'It makes optimization easier by controlling the scale of representations.\n\n'
        'But normalization is not just cosmetic. It changes the optimization landscape,\n'
        'and therefore what the network can learn in practice.',
        'It make optimization easier by control scale of representations.\n\n'
        'But normalization is not just cosmetic.\n'
        'It change optimization landscape, therefore what network can\n'
        '«learn practically».',
    ),
    (
        'D. Deep learning',
        'What happens during backpropagation?',
        'The loss tells us how wrong the final output is.\n'
        'Backpropagation moves that information backward through the computation graph\n'
        'using the chain rule.\n\n'
        'Each parameter gets a gradient: if I move this a little, which way does the loss move?',
        'Loss tell how wrong the final output is.\n'
        'Backprop move that information backward by chain rule.\n\n'
        'Each parameter get a gradient: if I move this a little,\n'
        '«which direction does the loss move?»',
    ),
    (
        'D. Deep learning',
        'What does the optimizer do?',
        'The gradient tells us the direction.\n'
        'The optimizer decides how to use that information to change the parameters.\n\n'
        'So backward does not train the model by itself. It computes information.\n'
        'The step actually changes the parameters.',
        'Gradient tell the direction.\n'
        'Optimizer decide how to use that information to change parameters.\n\n'
        'So backward does not train model by itself. It compute information.\n'
        '«Step actually changes the parameters.»',
    ),
    (
        'D. Deep learning',
        'What is your favorite optimizer?',
        'Usually AdamW for practical deep learning. But I have no religious attachment to it.\n\n'
        'The optimizer matters, but data, representation, architecture, and the learning-rate\n'
        'schedule can matter much more.',
        'Usually AdamW for practical deep learning. But no religious attachment.\n\n'
        'Optimizer is important, but data, representation, architecture,\n'
        'and learning-rate schedule can matter «much more».',
    ),
    (
        'E. Transformers',
        'Why are Transformers so successful?',
        'Attention gives a flexible way to decide which parts of the input should interact.\n\n'
        'More importantly, the same mechanism works across many domains.\n'
        'Language was the first huge success, but the mechanism itself is much more general.',
        'Attention give flexible way to decide which parts should interact.\n\n'
        'More important: same mechanism work across many domains.\n'
        'Language was first huge success, but mechanism itself is\n'
        '«much more general».',
    ),
    (
        'E. Transformers',
        'What is attention actually doing?',
        'Very roughly, every token asks: which other tokens are relevant to me?\n'
        'Then it collects information from those locations.\n\n'
        'The interesting part is that relevance is learned, not hard-coded.',
        'Very roughly, every token ask: which other tokens are relevant to me?\n'
        'Then it collect information from those locations.\n\n'
        'Interesting part: relevance is learned, «not hard-coded».',
    ),
    (
        'E. Transformers',
        'Why not just use full attention?',
        'Because the cost grows very badly with sequence length.\n\n'
        'For a huge spatial or temporal problem, full attention can become ridiculous.\n'
        'Then you need locality, windows, hierarchy, sparse attention -- something like that.',
        'Because cost grow very badly with sequence length.\n\n'
        'For huge spatial or temporal problem, full attention can become ridiculous.\n'
        'Then you need locality, windows, hierarchy, sparse attention --\n'
        '«something like that».',
    ),
    (
        'E. Transformers',
        'Why use CNNs if Transformers are better?',
        '"Better" depends on the structure.\n\n'
        'CNNs have a very strong local prior. If locality is real, I do not want to throw\n'
        'that prior away just because attention is fashionable.',
        '"Better" depend on the structure.\n\n'
        'CNNs have very strong local prior.\n'
        'If locality is real, I don\'t want throw that prior away\n'
        'just because attention is «fashionable».',
    ),
    (
        'E. Transformers',
        'What is your opinion about positional encoding?',
        'The model needs some way to know that two otherwise identical tokens are in\n'
        'different positions.\n\n'
        'But I am more interested in whether the positional representation matches the\n'
        'geometry of the problem. For geographic data, ordinary 1D PE is not necessarily right.',
        'Model need some way to know two otherwise identical tokens\n'
        'are in different positions.\n\n'
        'But I more interested whether positional representation match geometry of problem.\n'
        'For geographic data, ordinary 1D PE is «not necessarily right idea».',
    ),
    (
        'E. Transformers',
        'Why use RoPE?',
        'I like that relative position becomes important inside the attention calculation.\n\n'
        'For some problems, relative geometry is more natural than giving every position\n'
        'an unrelated embedding.',
        'I like that relative position become important\n'
        'inside the attention calculation.\n\n'
        'For some problems, relative geometry more natural\n'
        'than give every position an «unrelated embedding».',
    ),
    (
        'F. Generative / OOD',
        'What is a VAE useful for?',
        'I see it as a way to learn a structured latent space while keeping a generative\n'
        'relationship with the data.\n\n'
        'The important idea for me is not "generate pretty images."\n'
        'It is whether we can ask if a new observation belongs to the region the model learned.',
        'I see it as way to learn structured latent space\n'
        'while keep generative relationship with data.\n\n'
        'Important idea for me is not "generate pretty images."\n'
        'It is: can we ask whether new observation belong to\n'
        '«region the model has learned»?',
    ),
    (
        'F. Generative / OOD',
        'How would you detect an out-of-distribution input?',
        'I would not trust one detector.\n\n'
        'Reconstruction, density, ensemble disagreement, latent distance -- each tells us\n'
        'something different.\n\n'
        'I would combine several signals rather than pretend there is one perfect OOD score.',
        'I would not trust one detector.\n\n'
        'Reconstruction, density, ensemble disagreement, latent distance --\n'
        'each tell something different.\n\n'
        'I would combine several signals rather than pretend\n'
        'there is «one perfect OOD score».',
    ),
    (
        'F. Generative / OOD',
        'Can an autoencoder detect whether an input is familiar?',
        'It can give evidence, but not a proof.\n\n'
        'A powerful autoencoder can sometimes reconstruct strange things too well.\n'
        'So reconstruction error alone is not enough.\n'
        'I would look at the latent representation as well.',
        'It can give evidence, but not a proof.\n\n'
        'Powerful autoencoder can sometimes reconstruct strange things too well.\n'
        'So reconstruction error alone is not enough.\n'
        'I would look at latent representation «as well».',
    ),
    (
        'F. Generative / OOD',
        'What does a generative model learn?',
        'Ideally, some structure of the data distribution.\n\n'
        'But I am careful with the word "learn." It may memorize some things, interpolate\n'
        'some things, and discover useful regularities at the same time.',
        'Ideally, some structure of the data distribution.\n\n'
        'But I careful with word "learn."\n'
        'It may memorize some things, interpolate some things,\n'
        'and discover useful regularities «at the same time».',
    ),
    (
        'G. Debugging',
        'Your model isn\'t learning. What do you check?',
        'First, something embarrassingly simple: is the data correct?\n\n'
        'Then shapes, normalization, target alignment, loss scale, gradients, learning rate.\n\n'
        'I do not start redesigning the architecture. Very often the architecture is innocent.',
        'First, something embarrassingly simple: is the data correct?\n\n'
        'Then shapes, normalization, target alignment, loss scale,\n'
        'gradients, learning rate.\n\n'
        'I don\'t start redesign architecture.\n'
        'Very often architecture is «innocent».',
    ),
    (
        'G. Debugging',
        'Training loss decreases but validation loss doesn\'t. What do you do?',
        'First I verify the split and the data pipeline.\n'
        'If those are correct, then yes -- overfitting or a train/validation distribution gap.\n\n'
        'I would inspect examples, not only the curves.\n'
        'Curves tell me that something is happening; examples may tell me what.',
        'First verify the split and data pipeline.\n'
        'If those are correct, then yes -- overfitting or distribution difference.\n\n'
        'I would inspect examples, not only the curves.\n'
        'Curves tell me that something happen;\n'
        'examples may tell me «what».',
    ),
    (
        'G. Debugging',
        'How do you debug a neural network?',
        'I make the problem smaller until I can see it.\n\n'
        'One batch. Ten samples. Even one sample.\n\n'
        'If the model cannot overfit one tiny batch, I do not believe the large training run.',
        'I make the problem smaller until I can see it.\n\n'
        'One batch. Ten samples. Even one sample.\n\n'
        'If model cannot overfit one tiny batch,\n'
        'I don\'t believe the large training run. «This is my habit.»',
    ),
    (
        'G. Debugging',
        'How do you know whether your model is actually improving?',
        'I do not trust one metric.\n\n'
        'I want several views: loss, task metrics, examples, error distribution,\n'
        'and behavior on difficult cases.\n\n'
        'A single average number can hide almost anything.',
        'I don\'t trust one metric.\n\n'
        'I want several views: loss, task metrics, examples,\n'
        'error distribution, behavior on difficult cases.\n\n'
        'Single average number can hide «almost anything».',
    ),
    (
        'G. Debugging',
        'What is the most common ML mistake?',
        'I think it is solving the wrong problem very efficiently.\n\n'
        '(Then stop. Let them ask for an example.)\n\n'
        'Example: spend two months tuning a model when the target itself is poorly defined.\n'
        'The model is not the problem anymore.',
        'I think it is solving the wrong problem «very efficiently».\n\n'
        '(Then stop. Let them ask for example.)\n\n'
        'Example: spend two months tuning model when target itself\n'
        'is poorly defined. Model is not the problem «anymore».',
    ),
    (
        'H. Production',
        'What is different between research and production ML?',
        'In research, I can celebrate a one-percent improvement.\n'
        'In production, maybe that one percent is not worth doubling inference cost.\n\n'
        'Production has latency, reliability, monitoring, drift, rollback.\n'
        'The model is only one component.',
        'In research, I can celebrate one percent improvement.\n'
        'In production, maybe that one percent not worth doubling inference cost.\n\n'
        'Production has latency, reliability, monitoring, drift, rollback.\n'
        'Model is only «one component».',
    ),
    (
        'H. Production',
        'How do you monitor a model in production?',
        'Not only prediction accuracy. Accuracy may arrive too late because labels are delayed.\n\n'
        'I would monitor input distribution, missing values, feature stats, prediction\n'
        'distribution, latency, failures, and eventually the real outcome when available.',
        'Not only prediction accuracy.\n'
        'Accuracy may arrive too late because labels are delayed.\n\n'
        'I would monitor input distribution, missing values, feature stats,\n'
        'prediction distribution, latency, failures,\n'
        'and eventually real outcome when available.\n'
        '«Accuracy alone is too late.»',
    ),
    (
        'H. Production',
        'What is data drift?',
        'The distribution of what we receive has changed from what the model learned from.\n\n'
        'But drift itself does not necessarily mean the model is bad.\n'
        'The important question is whether the relationship between input and target changed.',
        'Distribution of what we receive has changed\n'
        'from what the model learned from.\n\n'
        'But drift itself don\'t necessarily mean model is bad.\n'
        'Important question: whether relationship between input and target\n'
        'has changed. «This distinction matters.»',
    ),
    (
        'H. Production',
        'How do you balance model accuracy and computational cost?',
        'I treat compute as part of the model design, not something we worry about at the end.\n\n'
        'If a model is 2% better but ten times more expensive, maybe it is not better.',
        'I treat compute as part of model design,\n'
        'not something we worry about at the end.\n\n'
        'If model is 2% better but ten times more expensive,\n'
        'maybe it is «not better».',
    ),
    (
        'I. Philosophical',
        'Do you think AI understands?',
        'It depends what we mean by understand.\n\n'
        'If understanding means a useful internal representation that supports prediction\n'
        'and reasoning, I think we should take the possibility seriously.\n\n'
        'If it means human subjective experience, that is a different question.',
        'It depend what we mean by understand.\n\n'
        'If understanding mean useful internal representation\n'
        'that support prediction and reasoning,\n'
        'I think we should take the possibility seriously.\n\n'
        'If it mean human subjective experience,\n'
        'that is «different question».',
    ),
    (
        'I. Philosophical',
        'Will scaling alone produce AGI?',
        'I do not know. I would not bet confidently either way.\n\n'
        'Scaling has produced capabilities people did not expect, so dismissing it seems\n'
        'dangerous. But I also do not know if more of the same is enough.\n\n'
        'Maybe there is a missing structural ingredient.',
        'I don\'t know. I would not bet confidently either way.\n\n'
        'Scaling produced capabilities people did not expect,\n'
        'so dismissing it seem dangerous.\n'
        'But I also don\'t know if more of the same is enough.\n\n'
        'Maybe there is «missing structural ingredient».',
    ),
    (
        'I. Philosophical',
        'What is the future of AI?',
        'The interesting direction is not simply bigger models.\n\n'
        'Models will become better at building internal representations of situations,\n'
        'maintaining state, interacting with the world, and knowing when they are uncertain.\n\n'
        'Language is one interface to intelligence. I do not think intelligence itself is language.',
        'Interesting direction is not simply bigger models.\n\n'
        'Models will become better at building internal representations,\n'
        'maintaining state, interacting with world,\n'
        'and knowing when they are uncertain.\n\n'
        'Language is one interface to intelligence.\n'
        'I don\'t think intelligence itself is language. «This is my bias.»',
    ),
]


def draw_qa_slide(step_no: int, section: str, question: str,
                  native: str, asian: str):
    asian_disp, twists = extract_twists(asian)
    native_disp = wrap_block(native, width=40)
    asian_disp = wrap_block(asian_disp, width=40)

    fig, ax = new_figure()
    fig.suptitle(
        f'Step {step_no}: {section}',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax)

    # Question
    round_box(ax, 0.3, 6.55, 11.4, 1.15, C_DARK, C_DARK, lw=0)
    ax.text(
        0.5, 7.35, 'Q',
        ha='left', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color='#f9e79f',
    )
    ax.text(
        1.15, 7.1, wrap_block(question, width=72),
        ha='left', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='white',
    )

    # Native panel
    round_box(ax, 0.3, 1.55, 5.55, 4.75, C_NATIVE, C_OK, lw=2.2)
    ax.text(
        3.05, 5.95, 'Native English',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color=C_OK,
    )
    ax.text(
        0.55, 3.7, native_disp,
        ha='left', va='center', fontsize=FS_ANNOT,
        color='#145a32', linespacing=1.28,
    )

    # Asian panel
    round_box(ax, 6.15, 1.55, 5.55, 4.75, C_ASIAN, '#d68910', lw=2.2)
    ax.text(
        8.9, 5.95, 'Asian English  (practice this)',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='#9a7d0a',
    )
    ax.text(
        6.4, 3.7, asian_disp,
        ha='left', va='center', fontsize=FS_ANNOT,
        color='#6e2c00', linespacing=1.28,
    )

    # Twist legend
    if twists:
        legend = 'Asian twists:  ' + '  |  '.join(
            f'[*{i}] «{t}»' for i, t in enumerate(twists, start=1)
        )
        legend = wrap_block(legend, width=100)
    else:
        legend = 'Asian twists:  (none marked on this slide)'

    round_box(ax, 0.3, 0.15, 11.4, 1.2, '#fadbd8', C_TWIST, lw=1.8)
    ax.text(
        6.0, 0.75, legend,
        ha='center', va='center', fontsize=FS_ANNOT - 1,
        color=C_TWIST, fontweight='bold',
    )

    show_maximized(fig)


# -------------------------------------------------------
# Step 0
# -------------------------------------------------------
fig0, ax0 = new_figure()
fig0.suptitle(
    'Step 0: Two voices per question -- Native vs Asian English',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax0)

round_box(ax0, 0.4, 6.25, 11.2, 1.35, C_DARK, C_DARK, lw=0)
ax0.text(
    6.0, 6.9,
    'Left = how a native might say it.\n'
    'Right = Asian English with more noise / grammar errors -- your practice target.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

cards = [
    (0.4, 3.35, C_NATIVE, C_OK, 'Native panel',
     'Clean articles, tense, agreement.\n'
     'Same technical idea.\n'
     'Use only as a reference,\n'
     'not as something to fake-fluently recite.'),
    (4.2, 3.35, C_ASIAN, '#d68910', 'Asian panel',
     'Missing a/the, topic-comment,\n'
     'verb forms, soft stops.\n'
     'More noise on purpose.\n'
     '«...» -> [*k] marks twists.'),
    (8.0, 3.35, C_GOAL, C_OK, 'Practice rule',
     'Read Q.\n'
     'Glance at Asian skeleton.\n'
     'Speak 15-25 seconds.\n'
     'Stop. Do not polish into essay English.'),
]
for x, y, face, edge, title, body in cards:
    round_box(ax0, x, y, 3.5, 2.55, face, edge, lw=2.0)
    ax0.text(x + 1.75, y + 2.1, title, ha='center',
             fontsize=FS_ANNOT + 1, fontweight='bold', color=edge)
    ax0.text(x + 1.75, y + 0.95, body, ha='center', va='center',
             fontsize=FS_ANNOT)

round_box(ax0, 0.4, 0.35, 11.2, 2.65, C_OLD, '#e67e22', lw=2.0)
ax0.text(
    6.0, 1.65,
    'Why two versions?\n'
    'Fluency can sound AI-copied in an interview.\n'
    'Clear ideas in irregular English can sound real -- and still technically sharp.\n\n'
    'Paradox: do not chase native fluency first. Chase clear thinking in speakable English.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig0)


# -------------------------------------------------------
# Steps 1..50
# -------------------------------------------------------
assert len(BANK) == 50, f'expected 50 Q&As, got {len(BANK)}'

for i, (section, question, native, asian) in enumerate(BANK, start=1):
    draw_qa_slide(i, section, question, native, asian)


# -------------------------------------------------------
# Step 51: Emergency phrases -- dual versions
# -------------------------------------------------------
figE, axE = new_figure()
figE.suptitle(
    'Step 51: Emergency phrases -- Native vs Asian',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(axE)

pairs = [
    ('Need time',
     'Let me think about this for a second.',
     'Let me think this for one second.'),
    ('Unclear Q',
     'Do you mean from the modeling side, or the production side?',
     'You mean modeling side, or production side?'),
    ('Stuck',
     'I know what I want to say, but I need a simpler way to say it.',
     'I know what I want say, but I need simpler way to say it.'),
    ('Only intuition',
     'This is more intuition than something I can prove right now.',
     'This more intuition than something I can prove right now.'),
    ('Disagree',
     'I am not completely convinced by that.',
     'I am not completely convince by that.'),
    ('Don\'t know',
     'I do not know the standard solution. I can tell you how I would investigate it.',
     'I don\'t know standard solution. I can tell how I would investigate.'),
]

axE.text(3.05, 7.35, 'Native', ha='center', fontsize=FS_ANNOT + 1,
         fontweight='bold', color=C_OK)
axE.text(8.7, 7.35, 'Asian', ha='center', fontsize=FS_ANNOT + 1,
         fontweight='bold', color='#9a7d0a')

for i, (tag, native, asian) in enumerate(pairs):
    y = 6.35 - i * 0.95
    round_box(axE, 0.25, y, 1.7, 0.8, C_MUTE, '0.45', lw=1.4)
    axE.text(1.1, y + 0.4, tag, ha='center', va='center',
             fontsize=FS_ANNOT - 1, fontweight='bold', color=C_DARK)
    round_box(axE, 2.05, y, 4.55, 0.8, C_NATIVE, C_OK, lw=1.4)
    axE.text(4.3, y + 0.4, wrap_block(native, width=36),
             ha='center', va='center', fontsize=FS_ANNOT - 1, color='#145a32')
    round_box(axE, 6.75, y, 4.9, 0.8, C_ASIAN, '#d68910', lw=1.4)
    axE.text(9.2, y + 0.4, wrap_block(asian, width=38),
             ha='center', va='center', fontsize=FS_ANNOT - 1, color='#6e2c00')

show_maximized(figE)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("011 ASIAN INTERVIEW SPEAKING BANK")
print("=" * 60)
print("""
Slides: Step 0 + Steps 1-50 (Q + Native A + Asian A) + Step 51

Each content slide:
  Left  = Native English (reference)
  Right = Asian English with more grammar noise (practice target)
  Red bar = «...» twists marked as [*k]

Practice:
  Speak from the Asian side. Use Native only to understand the intended meaning.
  Do not fake native fluency in the interview.
""")
