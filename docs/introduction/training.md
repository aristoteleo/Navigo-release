# Training

Navigo is trained on temporal populations **without known cell-to-cell lineages**. Let $X_i$ denote the population at time $t_i$. In principle, one would like to learn a vector field such that integrating from $X_i$ yields the next observed population $X_{i+1}$. However, **direct neural ODE training on large developmental atlases is computationally prohibitive**.

```{figure} ../_static/paper_figures/training_algo.png
:alt: Navigo flow-matching training panel
:width: 100%
:align: center

Population-level flow matching. Training alternates between pseudo-pair refinement and vector-field learning.
```

**Navigo adopts a flow matching strategy with iterative reflow**. The method first constructs pseudo-pairs between adjacent time points, then trains the vector field along interpolation paths defined by those pairs. If $\pi_i(j)$ denotes the paired cell index in $X_{i+1}$ for cell $x_j^{(i)} \in X_i$, the pairings are initialized randomly:

$$
\pi_i^{(0)}(j) \sim \mathrm{Uniform}\{1, \dots, |X_{i+1}|\}.
$$

For a pseudo-pair $(x_j^{(i)}, x_{j'}^{(i+1)})$ with $j' = \pi_i^{(k-1)}(j)$, Navigo defines a linear interpolation path and optimizes the field using the flow matching loss

$$
L^{(k)} =
\sum_{i=1}^{T-1}\sum_{j=1}^{|X_i|}
\mathbb{E}_{\tau \sim \mathrm{Uniform}[0,1]}
\left[
\left\|\Delta s_{ij} - (t_{i+1}-t_i) v_s(x_{j,j'}(t_i+\tau), t_i+\tau)\right\|_2^2
+
\left\|\Delta u_{ij} - (t_{i+1}-t_i) v_u(x_{j,j'}(t_i+\tau), t_i+\tau)\right\|_2^2
\right],
$$

where $\Delta s_{ij} = s_{j'}^{(i+1)} - s_j^{(i)}$ and $\Delta u_{ij} = u_{j'}^{(i+1)} - u_j^{(i)}$.

After fitting the field, the predicted next state is obtained by integrating from $t_i$ to $t_{i+1}$:

$$
\tilde{x}_j^{(i+1)} =
x_j^{(i)} + \int_{t_i}^{t_{i+1}} v^{(k)}(x_j(t), t)\,dt.
$$

The predicted state is then matched to the observed cells at the next time point through nearest-neighbor assignment:

$$
\pi_i^{(k)}(j) = \arg\min_{j'} \left\| \tilde{x}^{(i+1)}_j - x_{j'}^{(i+1)} \right\|_2^2.
$$

Training then alternates between fitting the field under the current pseudo-pairs and updating the pseudo-pairs using the new predictions. This iterative refinement is the reflow procedure.

To make the method practical for atlas-scale embryogenesis data, **Navigo also uses metacell aggregation before model training**. Cells are grouped within embryo, cell type, and time point; rare populations are preserved, while abundant populations are compressed. This substantially reduces computational cost while maintaining the main developmental structure needed for learning the vector field.

In this way, Navigo replaces intractable end-to-end trajectory supervision with an efficient flow-matching objective, while reflow improves temporal correspondence and metacell aggregation makes training feasible on large developmental datasets.
