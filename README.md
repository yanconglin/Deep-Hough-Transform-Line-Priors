# Hold on, I am editing right now...
# Deep-Hough-Transform-Line-Priors 
Official implementation for Deep-Hough-Transform-Line-Priors (ECCV 2020) 
https://arxiv.org/abs/2007.09493

Yancong Lin, and Silvia Laura Pintea, and Jan C. van Gemert

e-mail: y.lin-1ATtudelftDOTnl

Computer Vision Lab

Delft University of Technology, the Netherlands

## Introduction

Classical work on line segment detection is knowledge-based; it uses carefully designed geometric priors using either image gradients, pixel groupings, or Hough transform variants. Instead, current deep learning methods do away with all prior knowledge and replace priors by training deep networks on large manually annotated datasets. Here, we reduce the dependency on labeled data by building on the classic knowledge-based priors while using deep networks to learn features. We add line priors through a trainable Hough transform block into a deep network. Hough transform provides the prior knowledge about global line parameterizations, while the convolutional layers can learn the local gradient-like line features. On the Wireframe (ShanghaiTech) and York Urban datasets we show that adding prior knowledge improves data efficiency as line priors no longer need to be learned from data. Keywords: Hough transform; global line prior, line segment detection.

## Main feature: added Hough line priors

 <img src="ht-lcnn/figs/exp_gt.png" width="180">   <img src="ht-lcnn/figs/exp_input.png" width="180">   <img src="ht-lcnn/figs/exp_iht.png" width="180">   <img src="ht-lcnn/figs/exp_pred.png" width="180"> 
 
 From left to right:  Ground Truth, Input features, HTIHT output and Prediction 
 
## HT-IHT Module
 <img src="ht-lcnn/figs/htiht.png" width="480"> 
 
 The proposed HT-IHT module in our paper
