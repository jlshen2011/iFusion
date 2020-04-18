# *i*Fusion: Individualized Fusion Learning

This package implements my 2019 JASA paper - [*i*Fusion: Individualized Fusion Learning](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2019.1672557#.XciGbJJKg6U). 


## Introduction

*i*Fusion is a general statistical framework for making targeted inference and prediction. It is best suited in the scenario where:

* There is a number of **individual subject data**, which may or may not be generated from the same/similar data generating process. 
* One would like to make inference and prediction about a **target individual subject**. 


In general, this can be done by:

* Approach 1: Building a statistocal/machine learning model using the data associated with the target individual only (no/small bias, large variance);
* Approach 2: Or pooling the data for all individual subjects and building a single model without recognizing the potential heterogenities across different individuals (large bias, small variance).


*i*Fusion borrows information from others individual subjects (**fusion**), but in a smart way that only from individual subjects that are relevant to the target (**individualized**) , this optimizing the balance between bias and variance. 


<div align="center"><i>i</i>Fusion viewed from a bias-variance trade-off perspective</div>

<div align="center"><img src="images/idea.png?raw=true" width="500"/></div>


## Methodology

### The original *i*Fusion

The methodology in my original paper focuses on making individualized **inference** (that is, parameter estimation, confidence interval/region, hypothesis testing, etc.) with statistical guarantee, the package significantly simplifies the methods and shifts the emphasis towards **prediction**. In this section, I will give a quick review of the original *i*Fusion method. But feel free to skip to the next section. 


Inferences from different data sources can often be fused together, a process referred to as “fusion learning,” to yield more powerful findings than those from individual data sources alone. Effective fusion learning approaches are in growing demand as increasing number of data sources have become easily available in this big data era. 

This research proposes a new fusion learning approach, called “*i*Fusion,” for drawing efficient individualized inference by fusing learnings from relevant data sources. Specifically, *i*Fusion:	

1. summarizes inferences from individual data sources as individual confidence distributions (CDs); 
2. forms a clique of individuals that bear relevance to the target individual and then combines the CDs from those relevant individuals; 
3. draws inference for the target individual from the combined CD. 

In essence, iFusion strategically “borrows strength” from relevant individuals to enhance the efficiency of the target individual inference while preserving its validity. The research focuses on the setting where each individual study has a number of observations but its inference can be further improved by incorporating additional information from similar studies that is referred to as its clique. Under the setting, iFusion is shown to achieve oracle property under suitable conditions. It is also shown to be flexible and robust in handling heterogeneity arising from diverse data sources. The development is ideally suited for goal-directed applications. Computationally, iFusion is parallel in nature and scales up easily for big data. An efficient scalable algorithm is provided for implementation. 



<div align="center">How <i>i</i>Fusion works versus classical meta analysis inference</div>
<div align="center"><img src="images/flow.png?raw=true" width="600"/></div>





### The simpilified *i*Fusion and the ``ifusion`` package

<div align="center">Why <i>i</i>Fusion</div>

<img src="images/pros.png?raw=true" width="600"/>



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