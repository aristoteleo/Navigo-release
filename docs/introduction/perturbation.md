# Perturbation and GRN Analysis

**Navigo can simulate genetic perturbations** by modifying selected genes in the initial state and then integrating the learned developmental field forward while maintaining those perturbations. Given a target gene set $G$, the perturbed initial state is written as

$$
[x^{(i),\mathrm{pert}}_j]_g =
\begin{cases}
m \cdot [\bar{x}^{(i),c}]_g, & g \in G \\
[x^{(i)}_j]_g, & g \notin G ,
\end{cases}
$$

where $[\bar{x}^{(i),c}]_g$ is the mean expression of gene $g$ in the relevant cell type and $m$ controls perturbation strength. In this formulation, $m \leq 0$ corresponds to knockout-like perturbation and $m > 1$ corresponds to overexpression. Comparing perturbed and wild-type rollouts yields a **transcriptomic response profile** for the target gene or gene set.

The perturbed and wild-type trajectories are then defined as

$$
[x_j^{\mathrm{pert}}(t)]_g =
\begin{cases}
[x_j^{(i),\mathrm{pert}}]_g, & g \in G \\
[x_j^{(i),\mathrm{pert}}]_g + \int_{t_i}^{t} [v(x_j^{\mathrm{pert}}(\tau), \tau)]_g d\tau, & \text{otherwise},
\end{cases}
$$

and

$$
x_j^{\mathrm{WT}}(t) =
x_j^{(i)} + \int_{t_i}^{t} v(x_j^{\mathrm{WT}}(\tau), \tau)\,d\tau.
$$

For gene regulatory analysis, **Navigo uses the Jacobian of the learned transcription term**:

$$
J_{ij} = \frac{\partial \alpha_i}{\partial s_j}.
$$

which quantifies how the regulator-like expression $s_j$ affects the transcription rate of gene $i$ in the learned model. Because the Jacobian is defined from the transcription component itself, it provides a **local and interpretable view of regulatory influence** within the learned developmental field.

Together, perturbation rollout and Jacobian analysis extend Navigo beyond trajectory mapping. The learned field can be used to study lineage-specific regulatory programs, simulate knockout effects in a zero-shot setting, and evaluate candidate factors for cell-fate programming.
