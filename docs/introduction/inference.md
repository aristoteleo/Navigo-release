# Inference

After training, **the learned vector field can be used to predict cell states at specific developmental times through forward integration**. Starting from an observed cell state $x_j^{(i)}$ at time $t_i$, Navigo predicts the state at a later time $t_{i+m}$ by integrating

$$
\tilde{x}^{(i+m)}_j = x^{(i)}_j + \int_{t_i}^{t_{i+m}} v(x_j(t), t)\,dt.
$$

This is the **forward simulation** mode. It produces a predicted population by evolving the observed cells under the learned developmental field.

When observations are available at both flanking time points, Navigo also performs **aligned interpolation**. The model first simulates cells from the earlier time point to the later one:

$$
\tilde{x}^{(i+k)}_j = x^{(i)}_j + \int_{t_i}^{t_{i+k}} v(x_j(t), t)\,dt,
$$

then establishes correspondences by nearest-neighbor matching:

$$
\pi_i(j) = \arg\min_{j'} \left\| \tilde{x}^{(i+k)}_j - x_{j'}^{(i+k)} \right\|_2^2,
$$

and finally reconstructs the intermediate population by interpolation between aligned pairs:

$$
X_{i+m}^{\mathrm{aligned}} =
\left\{
\alpha x_j^{(i)} + (1-\alpha) x_{\pi_i(j)}^{(i+k)}
\right\}_{j=1}^{|X_i|},
\qquad
\alpha = \frac{k-m}{k}.
$$

The same field can also be used for **denoising**. In this setting, the model uses temporal continuity in the learned developmental process to recover a smoother and more biologically consistent population when the observed data at a given time point are noisy or under-sampled.

These inference procedures are all based on the same learned vector field. As a result, Navigo provides a unified framework for temporal prediction, interpolation, and denoising rather than a separate model for each task.
