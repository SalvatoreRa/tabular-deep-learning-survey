# Tabular Deep Learning: A Survey from Small Neural Networks to Large Language Models

This repository is the companion GitHub Project for [Tabular Deep Learning: A Survey from Small Neural Networks to Large Language Models](https://www.techrxiv.org/users/961472/articles/1332693-tabular-deep-learning-a-survey-from-small-neural-networks-to-large-language-models). The survey, will be published soon.

If you use any content of this repository or from the survey, please cite:



Feel free to create a new issue to signal any error, any suggestion, and any missing paper. A new version will be released soon. We will acknowledge any suggestion/correction.



# Index
* [Background](#Background)
* [Update](#Update)
* [How to cite](#How-to-cite)
* [Taxonomy](#Taxonomy)
    * [Data transformation for neural network](#Data-transformation-for-neural-network)
        * [Homogeneous Data Encoding](#Homogeneous-Data-Encoding)
    * [Neural ensembles](#Neural-ensembles)
    * [Regularization](#Regularization)
    * [Specialized architectures](#Specialized-architectures)
        * [MLP alternatives](#MLP-alternatives)
        * [MLP enhancements](#MLP-enhancements)
        * [Fully differentiable](#Fully-differentiable)
        * [Partly differentiable](#Partly-differentiable)
    * [Trasformer based](#Trasformer-based)
        * [Modified transformers](#Modified-transformers)
        * [Data adaptation](#Data-adaptation)
    * [Large language models](#Large-language-models)
    * [Benchmark and comparative studies](#Benchmark-and-comparative-studies)
* [Tabular data imputation](#Tabular-data-imputation)
* [Tabular data generation](#Tabular-data-generation)
* [Resources](#Resources)
    * [Other interesting surveys](#Other-interesting-surveys)
    * [Awesome GitHub Repositories](#Awesome-GitHub-Repositories)
    * [Tabular data libraries](#Tabular-data-libraries)
    * [Tutorials on tabular data](#Tutorials-on-tabular-data)
    * [Free courses](#Free-courses)
    * [Books](#Books)
    * [Master and PHD thesis](#Master-and-PHD-thesis)


# Background

![timeline of models](https://github.com/SalvatoreRa/tabular-deep-learning-survey/blob/main/images/timeline_tabular_deep.jpg)

Tabular data remains the most prevalent data format across industries such as finance, healthcare, and cybersecurity. Despite the widespread adoption of deep learning in image and text domains, tabular data tasks are still dominated by tree-based models like XGBoost and CatBoost. This survey explores the evolving landscape of deep learning for tabular data, offering a comprehensive and structured overview of the field.

From early MLPs to recent transformer-based architectures and the emergence of foundation models, we cover key challenges unique to tabular data, categorize state-of-the-art models, and provide practical insights into model selection, interpretability, and scalability.

# Update

* [9/25] - survey has been published

# How to cite

If this repository has been useful for your work, consider citing this repository:

BiB format for latex:

``` 
 @article{Raieli_2025,
title={Tabular Deep Learning: A Survey from Small Neural Networks to Large Language Models},
url={http://dx.doi.org/10.36227/techrxiv.175753732.26052568/v1},
DOI={10.36227/techrxiv.175753732.26052568/v1},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Raieli, Salvatore},
year={2025},
month=sep }
```

Chicago format:

``` 
Raieli, Salvatore. 2025. “Tabular Deep Learning: A Survey from Small Neural Networks to Large Language Models.” Institute of Electrical and Electronics Engineers (IEEE). September. https://doi.org/10.36227/techrxiv.175753732.26052568/v1
```


# Taxonomy

![taxonomy](https://github.com/SalvatoreRa/tabular-deep-learning-survey/blob/main/images/taxonomy.jpg)

## Data transformation for neural network

### Homogeneous Data Encoding

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | Tab2Visual | [Tab2Visual: Overcoming Limited Data in Tabular Data Classification Using Deep Learning with Visual Representations](https://arxiv.org/abs/2502.07181) | |
| 2024 | PTaRL | [Ptarl: Prototype-based tabular representation learning via space calibration](https://arxiv.org/abs/2407.05364) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/HangtingYe/PTaRL)  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Alcoholrithm/PTaRL)|
| 2024 | CARTE | [CARTE: pretraining and transfer for tabular learning](https://proceedings.mlr.press/v235/kim24d.html) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/soda-inria/carte) |
| 2024 | INCE | [Graph Neural Network contextual embedding for Deep Learning on tabular data](https://www.sciencedirect.com/science/article/pii/S0893608024001047?via%3Dihub) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MatteoSalvatori/INCE) |
| 2024 | LM-IGTD | [LM-IGTD: a 2d image generator for low-dimensional and mixed-type tabular data to leverage the potential of convolutional neural networks](https://arxiv.org/abs/2406.14566) | -  |
| 2023 | ReConTab | [Recontab: Regularized contrastive representation learning for tabular data](https://arxiv.org/abs/2310.18541) | - |
| 2023 | HYTREL | [HYTREL: Hypergraph-enhanced Tabular Data Representation Learninga](https://arxiv.org/abs/2307.08623) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/awslabs/hypergraph-tabular-lm) |
| 2023 | TabGSL | [TabGSL: Graph Structure Learning for Tabular Data Prediction](https://arxiv.org/abs/2305.15843) | - |
| 2023 | IGNNet  | [Interpretable Graph Neural Networks for Tabular Data](https://arxiv.org/abs/2308.08945) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/amrmalkhatib/ignnet) |
| 2023 | TabPTM | [Training-free generalization on heterogeneous tabular data via meta-representation](https://arxiv.org/abs/2311.00055) | - |
| 2022 | DWTM | [A Dynamic Weighted Tabular Method for Convolutional Neural Networks](https://arxiv.org/abs/2205.10386) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AnonymousCIKM1/DWTM) |
| 2022 | ReGram | [Look Around! A Neighbor Relation Graph Learning Framework for Real Estate Appraisal](https://arxiv.org/abs/2212.12190) | - |
| 2022 | T2G-Former | [T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction](https://arxiv.org/abs/2211.16887) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jyansir/t2g-former) |
| 2022 | Tab2Graph | [Tab2Graph: Transforming Heterogeneous Tables as Graphs](https://dl.acm.org/doi/full/10.1145/3703412.3703429) | - |
| 2021 | IGTD | [Converting tabular data into images for deep learning with convolutional neural networks](https://www.nature.com/articles/s41598-021-90923-y) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/oeg-upm/TINTOlib-Documentation) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/zhuyitan/IGTD) |
| 2020 | VIME | [VIME: extending the success of self- and semi-supervised learning to tabular domain](https://proceedings.neurips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jsyoon0823/VIME) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Alcoholrithm/TabularS3L)|
| 2020 | Tabular Convolution (TAC)  | [A novel method for classification of tabular data using convolutional neural networks](https://www.biorxiv.org/content/10.1101/2020.05.02.074203.abstract) | - |
| 2020 | REFINED | [Converting tabular data into images for deep learning with convolutional neural networks](https://www.nature.com/articles/s41598-021-90923-y) | - |
| 2019 | SuperTML | [Supertml: Two-dimensional word embedding for the precognition on structured tabular data](http://openaccess.thecvf.com/content_CVPRW_2019/html/Precognition/Sun_SuperTML_Two-Dimensional_Word_Embedding_for_the_Precognition_on_Structured_Tabular_CVPRW_2019_paper.html) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/oeg-upm/TINTOlib-Documentation)  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/EmjayAhn/SuperTML-pytorch)  |
| 2019 | DeepInsight | [DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture](https://www.nature.com/articles/s41598-019-47765-6) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://www.kaggle.com/code/markpeng/deepinsight-transforming-non-image-data-to-images) |



## Neural ensembles

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | TabM | [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabm) |
| 2025 |   Beta    | [Tabpfn unleashed: A scalable and effective solution to tabular classification problems](https://arxiv.org/abs/2502.02527) | - |
| 2025 | LLM-Boost, PFN-Boost | [Transformers Boost the Performance of Decision Trees on Tabular Data across Sample Sizes](https://arxiv.org/abs/2502.02672) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MayukaJ/LLM-Boost) |
| 2024 | HyperFast | [Hyperfast: Instant classification for tabular data](https://arxiv.org/abs/2402.14335) | AAAI        | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AI-sandbox/HyperFast) |
| 2023 | HyperTab | [HyperTab: Hypernetwork Approach for Deep Learning on Small Tabular Datasets](https://arxiv.org/abs/2304.03543) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/wwydmanski/hypertab) |


## Regularization

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | Harmonic loss| [Harmonic Loss Trains Interpretable AI Models](https://arxiv.org/abs/2502.01628v1) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ches-001/audio-segmenter)  |
| 2024 | sTabNet| [Escaping the Forest: Sparse Interpretable Neural Networks for Tabular Data](https://www.nature.com/articles/s44387-025-00056-0) [preprint](https://arxiv.org/abs/2410.17758v1) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/SalvatoreRa/sTabNet)  |
| 2023 | [An inductive bias for tabular deep learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html) | - |
| 2023 | [Self-supervision for Tabular Data by Learning to Predict Additive Homoskedastic Gaussian Noise as Pretext](https://dl.acm.org/doi/abs/10.1145/3594720) | - |
| 2023 | TANGOS | [Tangos: Regularizing tabular neural networks through gradient orthogonalization and specialization](https://arxiv.org/abs/2303.05506) |   [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/alanjeffares/TANGOS) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/vanderschaarlab/tangos) |
| 2022 | Regularized pretraining | [Revisiting Pretraining Objectives for Tabular Deep Learning](https://arxiv.org/abs/2207.03208) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/puhsu/tabular-dl-pretrain-objectives)  |
| 2022 | LSPIN| [Locally Sparse Neural Networks for Tabular Biomedical Data](https://proceedings.mlr.press/v162/yang22i.html) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jcyang34/lspin) |
| 2021 | Regularized tuned network| [Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/abs/2106.11189) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/machinelearningnuremberg/WellTunedSimpleNets)  |
| 2021 | Regularized cocktail| [Simple Modifications to Improve Tabular Neural Networks](https://arxiv.org/abs/2108.03214) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jrfiedler/xynn)  |
| 2021 | MLR | [Muddling Label Regularization: Deep Learning for Tabular Datasets](https://arxiv.org/abs/2106.04462) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/anonymousNeurIPS2021submission5254/SupplementaryMaterial)  |
| 2021 | LockOut| [Lockout: Sparse Regularization of Neural Networks](https://arxiv.org/abs/2107.07160) | - |
| 2019 | Cancelout| [CancelOut: A Layer for Feature Selection in Deep Neural Networks](https://dl.acm.org/doi/10.1007/978-3-030-30484-3_6) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/unnir/CancelOut)  |
| 2018 | STG| [Feature Selection using Stochastic Gates](https://arxiv.org/abs/1810.04247) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://yutaroyamada.com/stg/) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/runopti/stg) |
| 2018 | RLNs | [Regularization learning networks: deep learning for tabular datasets](https://proceedings.neurips.cc/paper_files/paper/2018/hash/500e75a036dc2d7d2fec5da1b71d36cc-Abstract.html) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/irashavitt/regularization_learning_networks) |
| 2017 | SNN| [Self-normalizing neural networks](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://www.kaggle.com/code/gulshanmishra/self-normalizing-neural-networks) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/bioinf-jku/SNNs) |



## Specialized architectures

### MLP alternatives

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2024 | KANs 2.0 | [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/kindxiaoming/pykan)  |
| 2024 | KANs | [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/kindxiaoming/pykan) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/kan-kolmogorov-arnold-networks) |
| 2024 |  ModernNCA | [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later](https://arxiv.org/abs/2407.03257) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/qile2000/LAMDA-TALENT) |
| 2022 | GANDALF | [GANDALF: Gated Adaptive Network for Deep Automated Learning of Features](https://arxiv.org/abs/2207.08548) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/manujosephv/GATE) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/manujosephv/pytorch_tabular) |
| 2022 |DANets| [Danets: Deep abstract networks for tabular data classification and regression](https://ojs.aaai.org/index.php/AAAI/article/view/20309) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/manujosephv/pytorch_tabular) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/WhatAShot/DANet) |
| 2016 | PNNs| [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Atomu2014/product-nets) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/product-based-neural-networks-for-user) |



### MLP enhancements

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2024 | sTabNet| [Escaping the Forest: Sparse Interpretable Neural Networks for Tabular Data](https://arxiv.org/abs/2410.17758v1) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/SalvatoreRa/sTabNet)  |
| 2024 | xNet | [Cauchy activation function and XNet](https://arxiv.org/abs/2409.19221) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/SalvatoreRa/tutorial/blob/main/artificial%20intelligence/FAQ.md#large-language-models:~:text=What%20are%20(Comple)XNet%20or%20Xnet%3F)  |
| 2024 | better by default| [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://arxiv.org/abs/2407.04491) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dholzmueller/pytabkit)  |
| 2021 | Regularized cocktail| [Simple Modifications to Improve Tabular Neural Networks](https://arxiv.org/abs/2108.03214) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jrfiedler/xynn)  |
| 2021 | Regularized cocktail| [Muddling Label Regularization: Deep Learning for Tabular Datasets](https://arxiv.org/abs/2106.04462) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/anonymousNeurIPS2021submission5254/SupplementaryMaterial)  |

### Fully differentiable

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2024 | BiSHop | [Bishop: Bi-directional cellular learning for tabular data with generalized sparse modern hopfield model](https://proceedings.mlr.press/v235/xu24l.html) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MAGICS-LAB/BiSHop) |
| 2024 |  GRANDE     | [GRANDE: gradient-based decision tree ensembles for tabular data](https://arxiv.org/abs/2309.17130) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/s-marton/GRANDE) |
| 2024 | SwitchTab | [Switchtab: Switched autoencoders are effective tabular learners](https://ojs.aaai.org/index.php/AAAI/article/view/29523) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Alcoholrithm/TabularS3L) |
| 2024 | ExcelFormer | [Can a deep learning model be a sure bet for tabular prediction?](https://dl.acm.org/doi/abs/10.1145/3637528.3671893) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/whatashot/excelformer) |
| 2023 | Trompt | [Trompt: Towards a better deep neural network for tabular data](https://proceedings.mlr.press/v202/chen23c.html) | - |
| 2021 | DNN2LR | [DNN2LR: Automatic Feature Crossing for Credit Scoring](https://arxiv.org/abs/2102.12036) | - |
| 2021 | SDTR | [SDTR: Soft Decision Tree Regressor for Tabular Data](https://ieeexplore.ieee.org/document/9393908) | - |
| 2021 | TabNet | [Tabnet: Attentive interpretable tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/16826) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dreamquark-ai/tabnet) |
| 2020 | NODE | [Neural oblivious decision ensembles for deep learning on tabular data](https://openreview.net/forum?id=r1eiu2VtwH) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/manujosephv/pytorch_tabular) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Qwicen/node) |
| 2020 | DCN V2 | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/dcn-m-improved-deep-cross-network-for-feature) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/tensorflow/recommenders/blob/v0.5.1/tensorflow_recommenders/layers/feature_interaction/dcn.py) |
| 2020 | NON | [Network On Network for Tabular Data Classification in Real-world Applications](https://arxiv.org/abs/2005.10114) | - |
| 2020 | DNF-Net | [DNF-Net: A Neural Architecture for Tabular Data](https://arxiv.org/abs/2006.06465) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/nini-lxz/DNF-Net) |
| 2018 | AutoInt| [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://deeptables.readthedocs.io/en/latest/) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/DeepGraphLearning/RecommenderSystems)  |
| 2018 | xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://deeptables.readthedocs.io/en/latest/) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Leavingseason/xDeepFM)  |
| 2017 | DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://deeptables.readthedocs.io/en/latest/) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/reczoo/FuxiCTR)  |
| 2017 | SDTS  | [Distilling a Neural Network Into a Soft Decision Tree](https://arxiv.org/abs/1711.09784) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/distilling-a-neural-network-into-a-soft)  |
| 2017 | DCN | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://deeptables.readthedocs.io/en/latest/) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/deep-cross-network-for-ad-click-predictions)  |
| 2016 | Wide & Deep | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792v1) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jrzaurin/pytorch-widedeep) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/wide-deep-learning-for-recommender-systems) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://deeptables.readthedocs.io/en/latest/) |

### Partly differentiable

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2023 | TabR | [TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023](https://arxiv.org/abs/2307.14338) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabular-dl-tabr) |
| 2021 | BGNN | [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://arxiv.org/abs/2101.08543) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/nd7141/bgnn) |
| 2019 | DeepGBM | [DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks](https://dl.acm.org/doi/10.1145/3292500.3330858) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/motefly/DeepGBM) |
| 2019 | TabNN | [TabNN: A Universal Neural Network Solution for Tabular Data](https://openreview.net/forum?id=r1eJssCqY7) | - |


## Trasformer based

### Modified transformers

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | TabPFN v2 | [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |
| 2024 | UniTabE | [UniTabE: A Universal Pretraining Protocol for Tabular Foundation Model in Data Science](https://openreview.net/forum?id=6LLho5X6xV) | |
| 2024 | MambaTab | [MambaTab: A Plug-and-Play Model for Learning Tabular Data](https://arxiv.org/abs/2401.08867) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/atik-ahamed/mambatab) |
| 2024 |  AMFormer | [Arithmetic feature interaction is necessary for deep tabular learning](https://arxiv.org/abs/2402.02334) |   [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/aigc-apps/AMFormer) |
| 2023 | TabPFN | [Tabpfn: A transformer that solves small tabular classification problems in a second](https://openreview.net/forum?id=cp5PvcI6w8_) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/PriorLabs/TabPFN) |
| 2023 |     TabRet     | [Tabret: Pre-training transformer-based tabular models for unseen columns](https://arxiv.org/abs/2303.15747) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/pfnet-research/tabret) |
| 2023 | Xtab | [Xtab: Cross-table pretraining for tabular transformers](https://proceedings.mlr.press/v202/zhu23k.html) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/BingzhaoZhu/XTab) |
| 2022 | LF-Transformer | [LF-Transformer: Latent Factorizer Transformer for Tabular Learning](https://ieeexplore.ieee.org/document/10401112) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/kwangtekNa/LF-Transformer/) |
| 2022 | TransTab| [Transtab: Learning transferable tabular transformers across tables](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1377f76686d56439a2bd7a91859972f5-Abstract-Conference.html) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://transtab.readthedocs.io/en/latest/) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/RyanWangZf/transtab) |
| 2022 | SAINTENS | [SAINTENS: Self-Attention and Intersample Attention Transformer for Digital Biomarker Development Using Tabular Healthcare Real World Data](https://pubmed.ncbi.nlm.nih.gov/35592984/) | - |
| 2022 | TableFormer | [TableFormer: Robust Transformer Modeling for Table-Text Encoding](https://arxiv.org/abs/2203.00274) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/google-research/tapas) |
| 2022 | SAINT | [SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training](https://openreview.net/forum?id=FiyUTAy4sB8) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/somepago/saint) |
| 2022 | GatedTabTransformer | [The GatedTabTransformer. An enhanced deep learning architecture for tabular modeling](https://arxiv.org/abs/2201.00199) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/radi-cho/GatedTabTransformer) |
| 2021 | NPT | [Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://arxiv.org/abs/2106.02584) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/OATML/Non-Parametric-Transformers) |
| 2021 | ARM-Net | [ARM-Net: Adaptive Relation Modeling Network for Structured Data](https://arxiv.org/abs/2107.01830) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/nusdbsystem/ARM-Net) |
| 2021 | FT-Transformer | [Revisiting deep learning models for tabular data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/rtdl) |
| 2021 | Fair-TabNet | [Fairness in TabNet Model by Disentangled Representation for the Prediction of Hospital No-Show](https://arxiv.org/abs/2103.04048) | - |
| 2021 | TabNet | [Tabnet: Attentive interpretable tabular learning](https://ojs.aaai.org/index.php/AAAI/article/view/16826) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dreamquark-ai/tabnet) |
| 2020 | TabTransformer | [Tabtransformer: Tabular data modeling using contextual embeddings](https://arxiv.org/abs/2012.06678) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/lucidrains/tab-transformer-pytorch) |
| 2020 | RPT | [RPT: Relational Pre-trained Transformer Is Almost All You Need towards Democratizing Data Preparation](https://arxiv.org/abs/2012.02469) | - |

### Data adaptation

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2022 | FORTAP | [FORTAP: Using Formulas for Numerical-Reasoning-Aware Table Pretraining](https://arxiv.org/abs/2106.03096) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/microsoft/TUTA_table_understanding) |
| 2021 | TabularNet | [TabularNet: A Neural Network Architecture for Understanding Semantic Structures of Tabular Data](https://arxiv.org/abs/2106.03096) | - |
| 2021 | DODUO | [Annotating Columns with Pre-trained Language Models](https://arxiv.org/abs/2104.01785) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/megagonlabs/doduo) |
| 2021 | MATE | [MATE: Multi-view Attention for Table Transformer Efficiency](https://arxiv.org/abs/2109.04312) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/google-research/tapas) |
| 2021 | TABBIE | [TABBIE: Pretrained Representations of Tabular Data](https://arxiv.org/abs/2105.02584) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/SFIG611/tabbie) |
| 2021 | DECO | [Exploring Decomposition for Table-based Fact Verification](https://arxiv.org/abs/2109.11020) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/arielsho/decomposition-table-reasoning) |
| 2020 | TAPAS | [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/google-research/tapas) |
| 2020 | TUTA | [TUTA: Tree-based Transformers for Generally Structured Table Pre-training](https://arxiv.org/abs/2010.12537) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/microsoft/TUTA_table_understanding) |
| 2020 | TaBERT | [TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data](https://arxiv.org/abs/2005.08314) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/facebookresearch/TaBERT) |
| 2020 | TURL | [TURL: Table Understanding through Representation Learning](https://arxiv.org/abs/2006.14806) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sunlab-osu/TURL) |
| 2020 | TableGPT | [TableGPT: Few-shot Table-to-Text Generation with Table Structure Reconstruction and Content Matching](https://aclanthology.org/2020.coling-main.179/) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/syw1996/TableGPT) |
| 2020 | ToTTo | [TToTTo: A Controlled Table-To-Text Generation Dataset](https://arxiv.org/abs/2004.14373) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/google-research-datasets/ToTTo) |
| 2019 | TabFact | [TabFact: A Large-scale Dataset for Table-based Fact Verification](https://arxiv.org/abs/1909.02164) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/wenhuchen/Table-Fact-Checking) |

## Large language models

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | Table-Critic | [Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](https://arxiv.org/abs/2502.11799v2) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Peiying-Yu/Table-Critic) |
| 2025 | NormTab | [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/soda-inria/tabicl) |
| 2025 | NormTab | [NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization](https://arxiv.org/abs/2406.17961) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/mahadi-nahid/NormTab) |
| 2024 | TabDPT | [TabDPT: Scaling Tabular Foundation Models on Real Data](https://arxiv.org/abs/2410.18164) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/layer6ai-labs/TabDPT-inference), [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/layer6ai-labs/TabDPT-training)|
| 2024 | SERVAL | [SERVAL: Synergy Learning between Vertical Models and LLMs towards Oracle-Level Zero-shot Medical Prediction](https://arxiv.org/abs/2403.01570v2) | - |
| 2024 | Table meets LLM | [Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study](https://arxiv.org/abs/2305.13062v5) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Y-Sui/Table-meets-LLM) |
| 2024 | PoTable | [PoTable: Towards Systematic Thinking via Stage-oriented Plan-then-Execute Reasoning on Tables](https://arxiv.org/abs/2412.04272) | - |
| 2024 | TART | [TART: An Open-Source Tool-Augmented Framework for Explainable Table-based Reasoning](https://arxiv.org/abs/2409.11724) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/xinyuanlu00/tart) |
| 2024 | TaPERA | [TaPERA: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning: Enhancing Faithfulness and Interpretability in Long-Form Table QA by Content Planning and Execution-based Reasoning](https://aclanthology.org/2024.acl-long.692/) | - |
| 2023 | TableGPT | [TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT](https://arxiv.org/abs/2307.08674) | - |
| 2023 | TAP4LLM | [TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning](https://arxiv.org/abs/2312.09039) | - |
| 2023 | TableLlama | [TableLlama: Towards Open Large Generalist Models for Tables](https://arxiv.org/abs/2311.09206) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://osu-nlp-group.github.io/TableLlama/)  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://huggingface.co/osunlp/TableLlama) [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/OSU-NLP-Group/TableLlama)|
| 2023 | UniTabPT | [Bridge the Gap between Language models and Tabular Understanding](https://arxiv.org/abs/2302.09302) | - |
| 2023 | UTP | [Testing the Limits of Unified Sequence to Sequence LLM Pretraining on Diverse Table Data Tasks](https://arxiv.org/abs/2310.00789) | - |
| 2023 | TABLET | [TABLET: Learning From Instructions For Tabular Data](https://arxiv.org/abs/2304.13188) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://paperswithcode.com/paper/tablet-learning-from-instructions-for-tabular) |
| 2023 | Summary boost | [Language models are weak learners](https://arxiv.org/abs/2306.14101) |  - |
| 2023 | TabFMs | [Towards Foundation Models for Learning on Tabular Data](https://openreview.net/forum?id=hz2zhaZPXm) |  - |
| 2023 | Unipredict | [UniPredict: Large Language Models are Universal Tabular Classifiers](https://arxiv.org/abs/2310.03266) |  - |
| 2023 | serializeLLM | [Towards Better Serialization of Tabular Data for Few-shot Classification with Large Language Models](https://arxiv.org/abs/2312.12464) |  - |
| 2023 | FinPT | [FinPT: Financial Risk Prediction with Profile Tuning on Pretrained Foundation Models](https://arxiv.org/abs/2308.00065) |  [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yuweiyin/finpt) |
| 2023 | DATER | [Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning](https://arxiv.org/abs/2301.13808) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/alibabaresearch/damo-convai) |
| 2023 | DocMath-Eval | [DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and Specialized Documents](https://arxiv.org/abs/2311.09805) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yale-nlp/docmath-eval) |
| 2023 | TableLLM | [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2312.16702) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Leolty/tablellm) |
| 2023 | MediTab | [MediTab: Scaling Medical Tabular Data Predictors via Data Consolidation, Enrichment, and Refinement](https://arxiv.org/abs/2305.12081) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ryanwangzf/meditab) |
| 2023 | ZeroTS | [Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ngruver/llmtime) |
| 2022 | Tabular representation | [Tabular Representation, Noisy Operators, and Impacts on Table Structure Understanding Tasks in LLMs](https://arxiv.org/abs/2310.10358) | - |
| 2022 | TabLLM | [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2210.10723) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/clinicalml/TabLLM) |
| 2022 | LIFT | [LIFT: Language-Interfaced Fine-Tuning for Non-Language Machine Learning Tasks](https://arxiv.org/abs/2206.06565) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/uw-madison-lee-lab/languageinterfacedfinetuning) |
| 2022 | OmniTab | [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/abs/2207.03637) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jzbjyb/omnitab) |
| 2022 | PromptCast | [PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/haounsw/pisa) |
| 2022 | FEVEROUS | [FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information](https://arxiv.org/abs/2106.05707) | [![Static Badge](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Raldir/FEVEROUS) |


## Benchmark and comparative studies

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2026 | Drug Response  | [Benchmarking community drug response prediction models: datasets, models, tools, and metrics for cross-dataset generalization analysis](https://academic.oup.com/bib/article/27/1/bbaf667/8422735) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/adpartin/cross-dataset-drp-paper/tree/v1.2-bib-paper-repro) |
| 2025 | TabArena  | [TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/autogluon/tabrepo?tab=readme-ov-file) |
| 2025 | TabReD  | [TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks](https://arxiv.org/abs/2406.19380) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tabred) |
| 2025 | Imputation benchmark  | [Imputation for prediction: beware of diminishing returns](https://arxiv.org/abs/2407.19804) |  - |
| 2024 | TALENT | [TALENT: A Tabular Analytics and Learning Toolbox](https://arxiv.org/abs/2407.04057) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/LAMDA-Tabular/TALENT?tab=readme-ov-file) |
| 2024 | Data-Centric Benchmark | [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/atschalz/dc_tabeval) |
| 2024 | Better-by-Default  | [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://arxiv.org/abs/2407.04491) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/dholzmueller/pytabkit) |
| 2024 | LAMDA-Tabular-Bench  | [A Closer Look at Deep Learning Methods on Tabular Datasets](https://arxiv.org/abs/2407.00956) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/qile2000/LAMDA-TALENT) |
| 2024 | DMLR-ICLR24 | [Towards Quantifying the Effect of Datasets for Benchmarking: A Look at Tabular Machine Learning](https://ml.informatik.uni-freiburg.de/wp-content/uploads/2024/04/61_towards_quantifying_the_effect.pdf) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/automl/dmlr-iclr24-datasets-for-benchmarking) |
| 2023 | TableShift  | [Benchmarking Distribution Shift in Tabular Data with TableShift](https://arxiv.org/abs/2312.07577) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/mlfoundations/tableshift) |
| 2023 | TabZilla  | [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/abs/2305.02997) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/naszilla/tabzilla) |
|   2023  |EncoderBenchmarking |[A benchmark of categorical encoders for binary classification](https://arxiv.org/abs/2307.09191)| [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/DrCohomology/EncoderBenchmarking)|
| 2022 | Grinsztajn - Benchmark | [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/LeoGrin/tabular-benchmark) |
| 2021 | RTDL | [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/rtdl-revisiting-models) |
| 2021 | WellTunedSimpleNets | [Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/abs/2106.11189) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/machinelearningnuremberg/WellTunedSimpleNets) |


## Tabular data imputation

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | TabDiff  | [TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation](https://arxiv.org/abs/2410.20626) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MinkaiXu/TabDiff) |
| 2025 | DeepIFSAC  | [DeepIFSAC: Deep Imputation of Missing Values Using Feature and Sample Attention within Contrastive Framework](https://arxiv.org/abs/2501.10910v3) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/mdsamad001/DeepIFSAC) |
| 2024 | demiss-vae  | [Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families](https://arxiv.org/abs/2403.03069) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/vsimkus/demiss-vae) |
| 2024 | NewImp | [Rethinking the Diffusion Models for Missing Data Imputation: A Gradient Flow Perspective](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cb1ba6a42814bf83974ed45ffdb72efa-Abstract-Conference.html) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/JustusvLiebig/NewImp) |
| 2024 | SimpDM   | [Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families](https://arxiv.org/abs/2407.18013) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yixinliu233/simpdm) |
| 2024 | DiffPuter  | [Unleashing the Potential of Diffusion Models for Incomplete Data Imputation](https://arxiv.org/abs/2405.20690) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/vsimkus/demiss-vae) |
| 2024 | MTabGen  | [Diffusion Models for Tabular Data Imputation and Synthetic Data Generation](https://arxiv.org/abs/2407.02549) | - |
| 2023 | Forest-Diffusion   | [Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees](https://arxiv.org/abs/2309.099687) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/SamsungSAILMontreal/ForestDiffusion) [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/atong01/conditional-flow-matching) |
| 2023 | MissDiff   | [MissDiff: Training Diffusion Models on Tabular Data with Missing Values](https://arxiv.org/abs/2307.00467) | - |
| 2023 | ReMasker   | [ReMasker: Imputing Tabular Data with Masked Autoencoding](https://arxiv.org/abs/2309.13793) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/tydusky/remasker) |
| 2023 | EGG-GAE   | [EGG-GAE: scalable graph neural networks for tabular data imputation](https://arxiv.org/abs/2210.10446) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/levtelyatnikov/EGG_GAE) |
| 2023 | WGAIN-GP   | [Enhanced data imputation framework for bridge health monitoring using Wasserstein generative adversarial networks with gradient penalty](https://www.sciencedirect.com/science/article/abs/pii/S2352012423013656) | - |
| 2022 | AimNet   | [Attention-based Learning for Missing Data Imputation in HoloClean](https://proceedings.mlsys.org/paper_files/paper/2020/hash/023560744aae353c03f7ae787f2998dd-Abstract.html) | - |
| 2022 | GEDI   | [GEDI: A Graph-based End-to-end Data Imputation Framework](https://arxiv.org/abs/2208.06573) | - |
| 2022 | TabCSDI  | [Diffusion models for missing value imputation in tabular data](https://arxiv.org/abs/2210.17128) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/pfnet-research/TabCSDI) |
| 2022 | B-VAEs  | [Leveraging variational autoencoders for multiple data imputation](https://arxiv.org/abs/2209.15321) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/roskamsh/BetaVAEMultImpute) |
| 2021 | NIMIWAE  | [Unsupervised Imputation of Non-ignorably Missing Data Using Importance-Weighted Autoencoders](https://arxiv.org/abs/2101.07357) |  - |
| 2021 | SGAIN  | [SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation](https://www.iccs-meeting.org/archive/iccs2021/papers/127420100.pdf) |  - |
| 2021 | DeepMVI  | [Missing Value Imputation on Multidimensional Time Series](https://arxiv.org/abs/2103.01600) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/eXascaleInfolab/bench-vldb20) |
| 2021 | PSMVAE  | [Deep Generative Pattern-Set Mixture Models for Nonignorable Missingness](https://arxiv.org/abs/2103.03532) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sghalebikesabi/PSMVAE) |
| 2019 | DataWig  | [DataWig: Missing Value Imputation for Tables](https://jmlr.org/papers/v20/18-753.html) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/awslabs/datawig) |
| 2019 | PPCA  | [Estimation and imputation in Probabilistic Principal Component Analysis with Missing Not At Random data](https://arxiv.org/abs/1906.02493) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/AudeSportisse/PPCA_MNAR) |
| 2019 | GP-VAE  | [GP-VAE: Deep Probabilistic Time Series Imputation](https://arxiv.org/abs/1907.04155) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ratschlab/GP-VAE) |
| 2019 | GINN  | [Missing Data Imputation with Adversarially-trained Graph Convolutional Networks](https://arxiv.org/abs/1905.01907) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/spindro/GINN) |
| 2018 | MIDA  | [MIDA: Multiple Imputation Using Denoising Autoencoders](https://link.springer.com/chapter/10.1007/978-3-319-93040-4_21) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Oracen-zz/MIDAS) |
| 2018 | GAIN  | [GAIN: Missing Data Imputation using Generative Adversarial Nets](https://arxiv.org/abs/1806.02920) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jsyoon0823/GAIN) |
| 2018 | HI-VAE  | [Handling Incomplete Heterogeneous Data using VAEs](https://arxiv.org/abs/1807.03653) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/probabilistic-learning/HI-VAE) |
| 2018 | DeepImpute  | ["Deep" Learning for Missing Value Imputationin Tables with Non-Numerical Data](https://dl.acm.org/doi/10.1145/3269206.3272005) |  - |
| 2012 | MissForest  | [MissForest—non-parametric missing value imputation for mixed-type data](https://academic.oup.com/bioinformatics/article/28/1/112/219101) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/stekhoven/missForest) |
| 2011 | MICE  | [Multiple imputation by chained equations: what is it and how does it work?](https://pubmed.ncbi.nlm.nih.gov/21499542/) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/amices/mice) |

## Tabular data generation

| Year | Name | Paper | Code | 
| - | - | - | - |
| 2025 | CDTD  | [Continuous Diffusion for Mixed-Type Tabular Data](https://arxiv.org/abs/2312.10431v4) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/muellermarkus/cdtd) |
| 2025 | TabDiff  | [TabDiff: a Unified Diffusion Model for Multi-Modal Tabular Data Generation](https://arxiv.org/abs/2410.20626) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/MinkaiXu/TabDiff) |
| 2024 | TabUnite  | [TabUnite: Efficient Encoding Schemes for Flow and Diffusion Tabular Generative Models](https://openreview.net/forum?id=Zoli4UAQVZ) | [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jacobyhsi/TabUnite) |
| 2024 | EntTabDiff  | [Entity-based Financial Tabular Data Synthesis with Diffusion Models](https://doi.org/10.1145/3677052.3698625) | - |
| 2024 | EHR-D3PM  | [Guided discrete diffusion for electronic health record generation](https://arxiv.org/abs/2404.12314) | - |
| 2024 | TabSyn  | [Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space](https://arxiv.org/abs/2310.09656) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/amazon-science/tabsyn) |
| 2024 | Forest-Diffusion  | [Generating and imputing tabular data via diffusion and flow-based gradient-boosted trees](https://arxiv.org/abs/2309.09968) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/SamsungSAILMontreal/ForestDiffusion) |
| 2023 | TabDDPM  | [Tabddpm: Modelling tabular data with diffusion models](https://arxiv.org/abs/2210.04018) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/yandex-research/tab-ddpm) |
| 2023 | GReaT  | [Language Models are Realistic Tabular Data Generators](https://arxiv.org/abs/2210.06280) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/tabularis-ai/be_great) |
| 2023 | REaLTabFormer  | [REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers](https://arxiv.org/abs/2302.02041) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/worldbank/REaLTabFormer) |
| 2023 | FinDiff  | [Findiff: Diffusion models for financial tabular data generation](https://arxiv.org/abs/2309.01472) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/sattarov/FinDiff) |
| 2023 | DPM-EHR  | [Synthetic health-related longitudinal data with mixed-type variables generated using diffusion models](https://arxiv.org/abs/2303.12281) |  - |
| 2023 | MedDiff | [MedDiff: Generating electronic health records using accelerated denoising diffusion model](https://arxiv.org/abs/2302.04355) |  - |
| 2023 | CoDi  | [Codi: Co-evolving contrastive diffusion models for mixed-type tabular synthesis](https://arxiv.org/abs/2304.12654) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/chaejeonglee/codi) |
| 2023 | AutoDiff  | [AutoDiff: combining Auto-encoder and Diffusion model for tabular data synthesizing](https://arxiv.org/abs/2310.15479) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/ucla-trustworthy-ai-lab/autodiffusion) |
| 2023 | STaSy  | [STaSy: Score-based Tabular data Synthesis](https://arxiv.org/abs/2210.04018) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/JayoungKim408/STaSy) |
| 2023 | MissDiff   | [MissDiff: Training Diffusion Models on Tabular Data with Missing Values](https://arxiv.org/abs/2307.00467) | - |
| 2022 | SOS  | [Sos: Score-based oversampling for tabular data](https://arxiv.org/abs/2206.08555) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/jayoungkim408/sos) |
| 2021 | SubTab  | [SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](https://arxiv.org/abs/2110.04361) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/astrazeneca/subtab) |
| 2021 | SCARF  | [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/abs/2106.15147) |  [![Code](https://badgen.net/badge/color/Code/black?label=)](https://github.com/Alcoholrithm/TabularS3L) |

# Resources

## Other interesting surveys

**2026**

* [Flow matching meets biology and life science: a survey](https://www.nature.com/articles/s44387-025-00066-y)
* [Generative Adversarial Networks for Synthetic Data Generation in Deep Learning Applications](https://www.artificialintelligencepub.com/jairi/article/view/jairi-aid1004)
* [AI-driven multi-omics integration in precision oncology: bridging the data deluge to clinical decisions](https://link.springer.com/article/10.1007/s10238-025-01965-9)

**2025**

* [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/abs/2504.16109)
* [A Survey on Deep Learning Approaches for Tabular Data Generation: Utility, Alignment, Fidelity, Privacy, Diversity, and Beyond](https://openreview.net/forum?id=RoShSRQQ67) - [preprint](https://arxiv.org/abs/2503.05954)
* [Tabular data generation models: An in-depth survey and performance benchmarks with extensive tuning](https://www.sciencedirect.com/science/article/abs/pii/S0925231225023276) - [preprint](https://hal.science/hal-04612244)
* [Survey on Tabular Data Privacy and Synthetic Data Generation in Industry 4.0](https://link.springer.com/article/10.1007/s10489-025-06823-5)
* [A Comprehensive Survey of Synthetic Tabular Data Generation](https://arxiv.org/abs/2504.16506)
* [A Survey of Large Language Models for Tabular Data Imputation: Tuning Paradigms and Challenges](https://link.springer.com/chapter/10.1007/978-3-032-03740-4_19)
* [Tabular Data Understanding with LLMs: A Survey of Recent Advances and Challenges](https://arxiv.org/abs/2508.00217)

**2024**

* [Language Modeling on Tabular Data: A Survey of Foundations, Techniques and Evolution](https://arxiv.org/abs/2408.10548)
* [Large Language Model for Table Processing: A Survey](https://arxiv.org/abs/2402.05121)
* [A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259)
* [Large Language Models(LLMs) on Tabular Data: Prediction, Generation, and Understanding -- A Survey](https://arxiv.org/abs/2402.17944)
* [Graph Neural Networks for Tabular Data Learning: A Survey with Taxonomy and Directions](https://arxiv.org/abs/2401.02143)
* [Deep Neural Networks and Tabular Data: A Survey](https://ieeexplore.ieee.org/abstract/document/9998482) - [code](https://github.com/kathrinse/TabSurvey?tab=readme-ov-file)


**2023**

* [Transformers for Tabular Data Representation: A Survey of Models and Applications](https://aclanthology.org/2023.tacl-1.14/)
* [Embeddings for Tabular Data: A Survey](https://arxiv.org/abs/2302.11777)


**2022**

* [Table Pre-training: A Survey on Model Architectures, Pre-training Objectives, and Downstream Tasks](https://arxiv.org/abs/2201.09745)

**2021**

* [Graph Signal Processing, Graph Neural Network and Graph Learning on Biological Data: A Systematic Review](https://ieeexplore.ieee.org/abstract/document/9585532)
* [Explainable Artificial Intelligence for Tabular Data: A Survey](https://ieeexplore.ieee.org/document/9551946)


## Awesome GitHub Repositories

* [Awesome Diffusion Models For Tabular Data](https://github.com/Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data?tab=readme-ov-file)
* [Awesome-Tabular-LLMs](https://github.com/SpursGoZmy/Awesome-Tabular-LLMs)
* [Awesome-LLM-Tabular](https://github.com/johnnyhwu/Awesome-LLM-Tabular)
* [Large Language Models on Tabular Data -- A Survey](https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation)
* [Awesome Tabular Data Augmentation](https://github.com/SuDIS-ZJU/awesome-tabular-data-augmentation)
* [Data Centric AI](https://github.com/HazyResearch/data-centric-ai)
* [Research on Tabular Deep Learning](https://github.com/yandex-research/rtdl)
* [Deep learning for tabular data](https://github.com/lmassaron/deep_learning_for_tabular_data)
* [Graph Neural Networks for Tabular Data Learning](https://github.com/roytsai27/awesome-gnn4tdl)
* [Machine Learning for Tabular Data](https://github.com/lmassaron/Machine-Learning-on-Tabular-Data)
* [Large Language Models on Tabular Data -- A Survey](https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation)

## Tabular data libraries

* [pytorch_tabular](https://github.com/manujosephv/pytorch_tabular) - Tabular deep learning models with PyTorch and PyTorch Lightening
* [Pytorch Frame](https://github.com/pyg-team/pytorch-frame) - A modular framework for building neural network models on heterogeneous tabular data.
* [TALENT](https://github.com/qile2000/LAMDA-TALENT)- A benchmark for tabular deep learning models.
* [DeepTables](https://github.com/DataCanvasIO/DeepTables) - a library for deep learning on tabula data
* [TabularS3L](https://github.com/Alcoholrithm/TabularS3L) - a PyTorch Lightning-based library designed to facilitate self- and semi-supervised learning with tabular data
* [mambular](https://mambular.readthedocs.io/en/latest/) - a Python library for tabular deep learning which includes Mamba, TabTransformer, FTTransformer, TabM and tabular ResNets, and others
* [TabPFN](https://github.com/PriorLabs/TabPFN) - TabPFN model and its [extensions](https://github.com/PriorLabs/tabpfn-extensions) 


## Tutorials on tabular data

From [here](), check there for other tutorials and resources

| Articles | notebook | description |
| ------- | ----------- | ------ |
|[Traditional ML Still Reigns: Why LLMs Struggle in Clinical Prediction?](https://levelup.gitconnected.com/traditional-ml-still-reigns-why-llms-struggle-in-clinical-prediction-0717b72bd37e) |--|Clinical prediction is more than medical knowledge: An LLM may not be the solution for every task |
|[Tabula Rasa: Why Do Tree-Based Algorithms Outperform Neural Networks](https://levelup.gitconnected.com/tabula-rasa-why-do-tree-based-algorithms-outperform-neural-networks-db641862859b)| -- |Tree-based algorithms are the winner in tabular data: Why?|
|[Tabula Rasa: How to save your network from the category drama](https://levelup.gitconnected.com/tabula-rasa-how-to-save-your-network-from-the-category-drama-623d67ad2e65)| -- | Neural networks do not like categories but you have techniques to save your favorite model |
|[Neural Ensemble: what’s Better than a Neural Network? A group of them](https://levelup.gitconnected.com/neural-ensemble-whats-better-than-a-neural-network-a-group-of-them-0c9e156fca15)| -- | Neural ensemble: how to combine different neural networks in a powerful model |
|[Tabula rasa: Give your Neural Networks Rules, They Will Learn Better](https://levelup.gitconnected.com/tabula-rasa-give-your-neural-networks-rules-they-will-learn-better-ba53b555cbb4)| -- |From great powers derive great responsibilities: regularization allows AI to exploit its power|
| [Tabula rasa: take the best of trees and neural networks](https://levelup.gitconnected.com/tabula-rasa-take-the-best-of-trees-and-neural-networks-ddc22c1884cb) | -- | Hybrid ideas for complex data: how to join two powerful models in one|
| [Tabula rasa: Could We Have a Transformer for Tabular Data](https://medium.com/gitconnected/tabula-rasa-could-we-have-a-transformer-for-tabular-data-9e4b238cde2c) | -- | We are using large language models for everything, so why not for tabular data? |
| [Tabula Rasa: not enough data? Generate them!](https://levelup.gitconnected.com/tabula-rasa-not-enough-data-generate-them-e1c160acb9c9) | -- | How you can apply generative AI to tabular data |
|[Tabula Rasa: Fill in What Is Missing](https://levelup.gitconnected.com/tabula-rasa-fill-in-what-is-missing-ba10c8402c03)| [Jupiter Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/tabula_rasa_missing_data.ipynb) - Scripts: [1](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/scripts/MAR.py), [2](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/scripts/MCAR.py), [3](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/scripts/MNAR.py) | Missing values are a known problem, why and how we can solve it|
| [Tabula Rasa: Large Language Models for Tabular Data](https://levelup.gitconnected.com/tabula-rasa-large-language-models-for-tabular-data-e1fd781946fa) | -- | Tabular data are everywhere, why and how you can use LLMs for them|
|[Tabula Rasa: A Deep Dive on Kolmogorov-Arnold Networks (KANs)](https://medium.com/p/f50958ca79b1)| -- |A Deep Dive into Next-Gen Neural Networks |

## Free courses

A list of free courses that can help to approach tabular deep learning

| Name | Link | Topic | Description |
| ------- | ------| ---- | ------ |
| 6.042J - Mathematics for Computer Science, Fall 2010, MIT OCW | [link](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-042j-mathematics-for-computer-science-fall-2010/video-lectures/) | Discrete Math | |
| 18.01 Single Variable Calculus, Fall 2006 - MIT OCW | [link](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/) | Calculus | |
| 18.02 Multivariable Calculus, Fall 2007 - MIT OCW | [link](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/) | Calculus  | |
| Statistics 110 - Probability - Harvard University | [link](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |  | |
| 18.06 - Linear Algebra, Prof. Gilbert Strang, MIT OCW | [link](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/) |  | |
| Causal inference| [link](https://www.bradyneal.com/causal-inference-course) | online causal inference course page |
| CS50: Introduction to Computer Science | [link](https://www.harvardonline.harvard.edu/course/cs50-introduction-computer-science) | introductory course  | Harvard introductory course. It’s an 11 week module based on Harvard University’s largest course called CS50. |
| Machine learning By Stanford| [link](https://cs229.stanford.edu/syllabus-autumn2018.html) | Machine learning| Machine Learning course taught by Andrew Ng at Stanford University |
| CS50 - Introduction to Artificial Intelligence with Python (and Machine Learning), Harvard OCW | [link](https://cs50.harvard.edu/ai/2020/) |  | |
| 6.034 Artificial Intelligence, MIT OCW | [link](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/) |  | |
| CS294-129 Designing, Visualizing and Understanding Deep Neural Networks | [link](https://bcourses.berkeley.edu/courses/1453965/pages/cs294-129-designing-visualizing-and-understanding-deep-neural-networks) |  | [YouTube](https://www.youtube.com/playlist?list=PLkFD6_40KJIxopmdJF_CLNqG3QuDFHQUm) |

## Books

* [Machine Learning for Tabular Data](https://github.com/lmassaron/Machine-Learning-on-Tabular-Data)
* [Deep Learning for Tabular Data](https://andre-ye.org/writing/mdl4td)
* [Deep Learning with Structured Data](https://www.manning.com/books/deep-learning-with-structured-data)

## Master and PHD thesis

**2026**

* [Learning with less: machine learning techniques for scarce medical data](https://www.repository.cam.ac.uk/items/f2ad2caa-5558-4343-8782-9c17082b101e)

**2025**

* [Exploring LLMs for Tabular Data Preparation](https://www.politesi.polimi.it/retrieve/079f58cb-758d-4b25-b379-97dee470ed2b/2025_04_Spreafico_Tassini_Tesi.pdf)
* [Mining Rules on Tabular Data](https://theses.hal.science/tel-05480277v1)
* [Transcriptomics data generation with deep generative models](https://theses.hal.science/tel-04996930v1)
* [Tabular Machine Learning on Small-Size and High-Dimensional Data](https://www.repository.cam.ac.uk/items/345cd12d-c798-425d-b885-633f80b081c1)
* [Toward effective and generalisable machine learning for biosignal time series](https://www.repository.cam.ac.uk/items/b99cb10e-b0ff-4820-bf2d-81ab3c8406ac)
* [On Principles of Efficiency for Federated Learning](https://www.repository.cam.ac.uk/items/f2d8af42-5228-42f9-8c52-149195c66be0)

**2024**

* [Reconciling deep learning with tabular data](https://theses.hal.science/tel-05035507v1)
* [Novel class discovery in tabular data : an application to network fault diagnosis](https://theses.hal.science/tel-04770701v1)
* [Advancing Anomaly Detection in Tabular Data : A Case-Study on Credit Card Fraud Identification](https://theses.hal.science/tel-05351694v1)

**2023**

* [Post-hoc Explainable AI for Black Box Models on Tabular Data](https://theses.hal.science/tel-04362470v1)
* [Deep Learning on Incomplete and Multi-Source Data : Application to Cancer Immunotherapy](https://theses.hal.science/tel-04804844v1)

**2022**

* [Deep learning models for tabular data curation](https://theses.hal.science/tel-03945305v1)
* [Tabular Data Integration for Multidimensional Data Warehouse](https://theses.hal.science/tel-03903570v1)
* [Deep learning for churn prediction](https://theses.hal.science/tel-04546983v1)

**2021**

* [Contributions to pattern set mining : from complex datasets to significant and useful pattern sets](https://theses.hal.science/tel-03342124v1)

**2020**

* [Deep learning for time series classification](https://theses.hal.science/tel-03715016v1)
