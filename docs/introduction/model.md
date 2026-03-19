# Model Principles

We follow the notation used in the manuscript. Let $T$ denote the number of observed developmental time points, with actual times $t_1, t_2, \dots, t_T$. For each time point $t_i$, let $X_i$ denote the observed cell population and let $x_j^{(i)}$ denote the $j$-th cell in that population.

For model training and inference, each cell is represented by its spliced and unspliced RNA state. We therefore write

$$
x = (s, u) \in \mathbb{R}^{2M},
$$

where $M$ is the number of genes, $s$ is the spliced RNA vector, and $u$ is the unspliced RNA vector.

**Navigo models development as a continuous vector field** over this state space:

$$
\frac{dx}{dt} = v(x, t).
$$

The objective is to learn a field that explains how the populations $\{X_i\}_{i=1}^T$ are connected through continuous developmental dynamics.

```{figure} ../_static/paper_figures/model_arch.png
:alt: Navigo model architecture from the paper
:width: 100%
:align: center

Model architecture from Figure 1e of the Navigo paper. The network predicts kinetic terms and converts them into developmental velocities.
```

Given a population $X_i$, the corresponding predicted next-time population is obtained by integrating the field:

$$
\tilde{X}_{i+1} =
\left\{
x_j^{(i)} + \int_{t_i}^{t_{i+1}} v(x_j(t), t)\,dt \;\middle|\; x_j^{(i)} \in X_i
\right\}.
$$

In principle, the field could then be learned by minimizing the distance between observed and predicted populations:

$$
\min_v \sum_{i=1}^{T-1} \mathrm{Distance}\left(X_{i+1}, \tilde{X}_{i+1}\right).
$$

This motivates the training formulation.

At the molecular level, the vector field network takes the spliced and unspliced RNA state and predicts the kinetic quantities $\alpha$, $\beta$, and $\gamma$. These determine the velocity components through RNA kinetics:

$$
v_u = \alpha(s, t) - \beta \odot u, \qquad
v_s = \beta \odot u - \gamma \odot s.
$$

Here $\alpha(s,t)$ is the transcription rate, modeled as a function of the mature RNA state, while $\beta$ and $\gamma$ are the splicing and degradation rates. In this formulation, the transcription term is cell-specific and gene-specific, whereas the splicing and degradation terms are gene-specific kinetic parameters.

At the population level, the learned field provides a continuous description of development across discrete temporal snapshots. This is essential because scRNA-seq atlases observe populations at sampled time points but **do not provide true biological correspondences between individual cells across time**. Navigo therefore combines a **population-level generative model** with a **molecular-level kinetic model** in a single architecture.
