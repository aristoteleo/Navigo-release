# Introduction

Embryonic development is orchestrated by complex gene regulatory networks, and learning the regulatory dynamics from developmental data could enable us to understand, predict, and ultimately program cell fates. Here we introduce Navigo, a biologically grounded generative modeling framework that learns a developmental vector field by integrating flow matching at the population level with RNA kinetics modeling at the molecular level. Navigo accurately maps developmental trajectories across lineages on a mouse embryogenesis scRNA-seq atlas spanning forty time points with ten million cells. Applied to cardiac development, Navigo enables mechanistic understanding of regulatory networks distinguishing congenital heart disease subtypes. Navigo's zero-shot predictive capability is validated on independent \textit{in vivo} knockout data for six genes without requiring perturbation-specific training, uncovering novel insights on lineage-specific gene compensation mechanisms. Moreover, Navigo enables rational programming of cell fates exemplified by fibroblast reprogramming, including identifying pro-fibrotic barriers to cardiac fates and evaluating hundreds of synergistic transcription factor combinations within gene families for neuronal fates. Overall, Navigo enables predictive and programmable generative modeling of developmental processes, advancing AI-based virtual embryos for developmental biology and regenerative medicine.

**Navigo is a generative model for developmental single-cell transcriptomics.** The core technical goal is to turn a temporal atlas of static scRNA-seq snapshots into a **continuous developmental vector field** that can be queried for trajectory mapping, temporal interpolation, denoising, perturbation simulation, and regulatory analysis.

The method is built around two ideas from the Navigo manuscript:

1. Development should be modeled at the **molecular level** using RNA kinetics, so the learned velocity field is tied to transcription, splicing, and degradation rather than an arbitrary black-box transition function.
2. Development should also be modeled at the **population level**, where the system must learn coherent transitions between time points even though destructive single-cell measurements do not provide true cell-to-cell correspondences across time.

This section introduces the technical principles behind that design, with emphasis on **four key parts**:

- **Model principles**: how Navigo represents cell state and defines the developmental vector field through RNA kinetics.
- **Training**: how flow matching, reflow, and metacell aggregation make the model trainable on atlas-scale data.
- **Inference**: how the learned vector field is used for forward simulation, aligned interpolation, and denoising.
- **Perturbation and GRN Analysis**: how the same model supports in silico perturbation and Jacobian-based regulatory interpretation.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Model Principles
:link: model
:link-type: doc

How Navigo represents cell state, parameterizes the developmental vector field, and connects RNA kinetics to gene regulation.
:::

:::{grid-item-card} Training
:link: training
:link-type: doc

How flow matching and metacell aggregation make training feasible on atlas-scale data.
:::

:::{grid-item-card} Inference
:link: inference
:link-type: doc

How the learned field is used for forward simulation, aligned interpolation, and denoising at specific developmental times.
:::

:::{grid-item-card} Perturbation and GRN Analysis
:link: perturbation
:link-type: doc

How Navigo performs in silico perturbation, zero-shot knockout prediction, and Jacobian-based regulatory analysis.
:::
::::

```{toctree}
:maxdepth: 2

model
training
inference
perturbation
```
