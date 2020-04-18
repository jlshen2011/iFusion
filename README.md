# *i*Fusion: Individualized Fusion Learning

This package implements my 2019 JASA paper - [*i*Fusion: Individualized Fusion Learning](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2019.1672557#.XciGbJJKg6U). 


## Table of Contents

  * [Introduction](#Introduction)
  * [Methodology](#screenshot)
  	* [The original *i*Fusion](#original)
    * [The simplified *i*Fusion and the ``ifusion`` package](#simplified)  
  * [Installation](#installation)
  * [Documentation](#documentation)  
  * [References](#references)
  	* [Main article](#main)
    * [More references](#more)  


<a name="introduction"></a>
## Introduction

*i*Fusion is a general statistical framework for making targeted inference and prediction. It is best suited in the scenario where:

* There is a number of **individual subject data**, which may or may not be from the same/similar data generating processes. 
* One would like to make inference and prediction about a **target individual subject**. 


In general, this can be done by:

* **Approach 1**: Building a statistical/machine learning model using the data associated with the target individual only (no/small bias, large variance);
* **Approach 2**: Pooling the data for all individual subjects and building a single model without recognizing the potential heterogeneities across different individuals (large bias, small variance).

*i*Fusion borrows information from others individual subjects (**fusion**), but in a smart way that only from individual subjects that are relevant to the target (**individualized**) , this optimizing the balance between bias and variance. 


<div align="center"><i>i</i>Fusion viewed from a bias-variance trade-off perspective</div>
<div align="center"><img src="images/idea.png?raw=true" width="500"/></div>
<br></br>


<a name="methodology"></a>
## Methodology

<a name="original"></a>
### The original *i*Fusion

Inferences from different data sources can often be fused together, a process referred to as “fusion learning,” to yield more powerful findings than those from individual data sources alone. Effective fusion learning approaches are in growing demand as increasing number of data sources have become easily available in this big data era. *i*Fusion fits into the fusion learning framework, but has a focus on making efficient **individualized** inference. Specifically, *i*Fusion:	

1. summarizes inferences from individual data sources as individual confidence distributions (CDs; roughly speaking, a CD is a distribution estimate for a parameter of interest with statistical guarantee, in contrast to a point estimate or a interval estimate; for those familiar with Bayesian statistics, you may think CD as a posterior distribution of a parameter but without any prior included).
2. forms a clique of individuals that bear relevance to the target individual and then combines the CDs from those relevant individuals. How to combines the CDs is to the key of *i*Fusion. At high level, it adaptively constructs a weight vector that measures the relevance of each individual subject to the target individual subject based on the individual CDs, and then apply a formula with this weight vector to obtain a combined CD. 
3. draws inference for the target individual from the combined CD. 

<div align="center">How <i>i</i>Fusion works versus classical meta-analysis inference</div>
<div align="center"><img src="images/flow.png?raw=true" width="600"/></div>
<br></br>


In essence, *i*Fusion strategically “borrows strength” from relevant individuals to enhance the efficiency of the target individual inference while preserving its validity. The research focuses on the setting where each individual study has a number of observations but its inference can be further improved by incorporating additional information from similar individual subjects. Under the setting, *i*Fusion is shown to achieve oracle property under suitable conditions. The following figure highlights some nice features about *i*Fusion. 

<div align="center">Why <i>i</i>Fusion?</div>
<div align="center"><img src="images/pros.png?raw=true" width="600"/></div>
<br></br>

<a name="simplified"></a>
### The simplified *i*Fusion and the ``ifusion`` package

The methodology in my original paper focuses on making individualized **inference** with statistical guarantee, that is, things like parameter estimation, confidence interval/region, hypothesis testing, and so on. Improved prediction is a byproduct of improved inference.

Because the goal is inference in the original *i*Fusion, the combination of individual CDs relies on a sophisticated algorithm. In this package, however, I will simplify the method significantly by having the goal to make efficient predictions about the target individual subject - many of the practitioners might be more interested in. 

In particular, the simplified method:
1. train a individual learner (in principal, any supervised learning model) for each individual subject data. 
2. Use the individual learners to make prediction on the target individual data; for the target individual itself, rather than using the individual learner trained on its full data, use cross-validation framework to make an "out-of-sample" prediction for each of the target individual's own data. 
3. Now, for each point in the target individual subject data, we will have *K* (the number of individual subjects) "out-of-sample" predictions for it. Use these predictions are feature variables to train a fusion learner. 
4. For a new data point for the target individual subject with unknown response, first use the *K* individual learners to obtain *K* predictions, and then use the fusion learner to make a final prediction for it. 


<a name="installation"></a>
## Installation

```bash
pip install ifusion
```

<a name="documentation"></a>
## Documentation


<a name="references"></a>
## References

<a name="main"></a>
### Main article
- [Full article]: **Jieli Shen**, Minge Xie, and Regina Liu. (2019). [*i*Fusion: Individualized Fusion Learning](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2019.1672557#.XciGbJJKg6U). *Journal of the American Statistical Association*, to appear.

<a name="more"></a>
### More references
- [My PhD thesis]: Jieli Shen. (2017). [Advances in confidence distribution: individualized fusion learning and predictive distribution function](https://rucore.libraries.rutgers.edu/rutgers-lib/55689/). PhD Thesis.
- [Review article on confidence distribution]: Minge Xie, and Kesar Singh. (2013). [Confidence distribution, the frequentist distribution estimator of a parameter: A review](https://www.stat.rutgers.edu/home/mxie/RCPapers/insr.12000.pdf). *International Statistical Review*, **81**, 3–39.
