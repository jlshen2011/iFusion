# *i*Fusion: Individualized Fusion Learning

This package implements my 2019 JASA paper - [*i*Fusion: Individualized Fusion Learning](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2019.1672557#.XciGbJJKg6U). 


# Table of Contents
=================

  * [Introduction](## Introduction)
  * [Screenshot](#screenshot)
  * [Installation](#installation)
        * [OR using Pathogen:](#or-using-pathogen)
        * [OR using Vundle:](#or-using-vundle)
  * [License](#license)


## Introduction(#Introduction)

*i*Fusion is a general statistical framework for making targeted inference and prediction. It is best suited in the scenario where:

* There is a number of **individual subject data**, which may or may not be from the same/similar data generating processes. 
* One would like to make inference and prediction about a **target individual subject**. 


In general, this can be done by:

* **Approach 1**: Building a statistocal/machine learning model using the data associated with the target individual only (no/small bias, large variance);
* **Approach 2**: Pooling the data for all individual subjects and building a single model without recognizing the potential heterogenities across different individuals (large bias, small variance).

*i*Fusion borrows information from others individual subjects (**fusion**), but in a smart way that only from individual subjects that are relevant to the target (**individualized**) , this optimizing the balance between bias and variance. 


<div align="center"><i>i</i>Fusion viewed from a bias-variance trade-off perspective</div>

<div align="center"><img src="images/idea.png?raw=true" width="500"/></div>


## Methodology

### The original *i*Fusion

The methodology in my original paper focuses on making individualized **inference** (that is, parameter estimation, confidence interval/region, hypothesis testing, etc.) with statistical guarantee, the package significantly simplifies the methods and shifts the emphasis towards **prediction**. In this section, I will give a quick review of the original *i*Fusion method. But feel free to skip to the next section. 


Inferences from different data sources can often be fused together, a process referred to as “fusion learning,” to yield more powerful findings than those from individual data sources alone. Effective fusion learning approaches are in growing demand as increasing number of data sources have become easily available in this big data era. *i*Fusion fits into the fusion learning framework, but has a focus on making efficient **individualized** inference. Specifically, *i*Fusion:	

1. summarizes inferences from individual data sources as individual confidence distributions (CDs; roughly speaking, a CD is a distribution estimate for a parameter of interest with statistical gurantee, in contrast to a point estimate or a interval estimate; for those familiar with Bayesian statistics, you may think CD as a posterior distribution of a parameter but without any prior included).
2. forms a clique of individuals that bear relevance to the target individual and then combines the CDs from those relevant individuals. How to combines the CDs is to the key of *i$Fusion. At high level, it first constructs a weight vector that measures the relevance of each individual subject to the target individual subject based on the individual CDs, and then apply a formula with this weight vector to obtain a combined CD. 
3. draws inference for the target individual from the combined CD. 

<div align="center">How <i>i</i>Fusion works versus classical meta analysis inference</div>
<div align="center"><img src="images/flow.png?raw=true" width="600"/></div>


In essence, *i*Fusion strategically “borrows strength” from relevant individuals to enhance the efficiency of the target individual inference while preserving its validity. The research focuses on the setting where each individual study has a number of observations but its inference can be further improved by incorporating additional information from similar individual subjects. Under the setting, iFusion is shown to achieve oracle property under suitable conditions. The following figure highlights some nice features about *i*Fusion. 

<div align="center">Why <i>i</i>Fusion?</div>
<div align="center"><img src="images/pros.png?raw=true" width="600"/></div>


### The simpilified *i*Fusion and the ``ifusion`` package




## Installation

```bash
pip install ifusion
```

## Documentation



## References

### Main articles
- [Full article]: **Jieli Shen**, Minge Xie, and Regina Liu. (2019). [*i*Fusion: Individualized Fusion Learning](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2019.1672557#.XciGbJJKg6U). *Journal of the American Statistical Association*, to appear.


### More references
- [My PhD thesis]: Jieli Shen. (2017). [Advances in confidence distribution: individualized fusion learning and predictive distribution function](https://rucore.libraries.rutgers.edu/rutgers-lib/55689/). PhD Thesis.
- [Review article on confidence distribution]: Minge Xie, and Kesar Singh. (2013). [Confidence distribution, the frequentist distribution estimator of a parameter: A review](https://www.stat.rutgers.edu/home/mxie/RCPapers/insr.12000.pdf). *International Statistical Review*, **81**, 3–39.