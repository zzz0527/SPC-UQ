### Toy Example: SPC-UQ on Regression

This is the toy example for SPC-UQ in the regression task.

To intuitively illustrate the regression setting, we construct a synthetic cubic regression task with zero-mean, asymmetric log-normal noise. The training samples are drawn from the function:

    y = x³ + ε(x) - E[ε(x)],  where  ε(x) ~ LogNormal(1.5, 1)

The input `x` is uniformly sampled from the range [-4, 4].  
The test set is similarly constructed, but with inputs sampled from a broader range [-6, 6].

- Training set: 2,000 samples  
- Test set: 1,000 samples  
- In-distribution (iD): x ∈ [-4, 4]  
- Out-of-distribution (OOD): x ∈ [-6, -4) ∪ (4, 6]

This setup introduces asymmetric and heteroscedastic noise to better evaluate the model’s ability to capture both aleatoric and epistemic uncertainty.
