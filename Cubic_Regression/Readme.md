This is the toy example for the SPC_UQ in the regression task.


For an intuitive illustration of the regression setting, we construct a synthetic cubic regression task with zero-mean, asymmetric log-normal noise. The training samples are drawn from the function:
\[
y = x^3 + \epsilon(x) - \mathbb{E}[\epsilon(x)], \quad \epsilon(x) \sim \mathrm{LogNormal}(1.5, 1).
\]
Where the input \( x \) is sampled uniformly from the range \( [-4, 4] \). The test set is similarly constructed, with inputs sampled from a broader range \( [-6, 6] \). The training set contains 2,000 data points, while the test set consists of 1,000 samples. We define the interval \( [-4, 4] \) as the in-distribution (iD) region, while the regions outside this interval, i.e., \( [-6, -4) \cup (4, 6] \), are considered as out-of-distribution (OOD) region. 
