# The ***``Browser Bench Analysis``***

A numerical analysis of 20 browsers in 5 (+2) performance tests.

## 1. Introduction

In choosing a web browser to use, one can assume any preffered criterion: familiarity, experience, features, usability, etc. One possible criterion is browser performance that can be measured by many tests assessing different factors of browser activity. The youtube channel Diolinux, at the very last days of december 2022, sucssesfully atempts to provide information about that topic, cataloging the results of 20 browsers undergoing five well known performance tests and, plus, two idealized by Diolinux itself. Diolinux does not gives any answer to what browser would be the more performatic for the year of 2023, instead he provides very elucidating graphs comparing all browsers in each tests. This notebook aims to provide an unique or multiple answer to the question: "based solely on performance, what would be the best browser to use?"

In order to answer this question this project aims to use unsupervised and non-parametric techiniques to: i) diagnose the behavior of the dataset features; and ii) assess the best performing browser. The first can be achieved with the use of *``Principal Components Analysis (PCA)``*, a non-supervised machine learning techique that summarizes the features, variables, of a dataset to a certain number of factors that come from the latent linear combinations for all variables. It is like producing a general grade by grouping STEM subjects (as maths and physics) apart from social studies (history and politics)

The *``Data Envelopment Analysis (DEA)``* can be used for the item ii. DEA is a non-parametric technique that uses linear programming in order to create efficiency scores for each one of its *``Decision Making Units (DMUs)``* - the individuals responsibles to performs a certain activity in the most efficient, costless, way. In this particular case, after carefully analyzing the dataset, one can assume each browser as one DMU for evaluating its performance based on tests results

## 2. Updates

* **README minor fixes**

## 3. Backlog

### RELEASE 1

1. [X] Data handling
2. [X] Exploratory data analysis
3. [X] Shallow PCA
4. [X] PCA tools construction
5. [X] Shallow PCA docs
6. [X] Shallow docs

### RELEASE 2

7. [ ] Deep PCA
8. [ ] PCA-DEA validation
9. [ ] DEA modeling
10. [ ] Deep PCA docs

### RELEASE 3

11. [ ] Shallow DEA
12. [ ] DEA tools construction
13. [ ] Shallow DEA docs

### RELEASE 4

14. [ ] Deep DEA
15. [ ] DEA validation
16. [ ] Deep DEA docs

### RELEASE 5

17. [ ] Deep docs
18. [ ] Docs editing