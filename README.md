# Beyond Correlation: Causal Inference

[![Contributors][contributors-shield]][contributors-url][![Forks][forks-shield]][forks-url][![Stargazers][stars-shield]][stars-url][![Issues][issues-shield]][issues-url][![MIT License][license-shield]][license-url][![LinkedIn][linkedin-shield]][linkedin-url]

![Causal-Graph][readme-image]
## Description

This project focusses on using causal inference to answer causal questions related to breast cancer cases. It uses a tabular data with features about cell samples of different individuals. The data used in this project is taken from [kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). It can also be downloaded from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets?name=breast).

A common frustration in the industry, especially when it comes to getting business insights from tabular data, is that the most interesting questions (from their perspective) are often not answerable with observational data alone. These questions can be similar to:
>“What will happen if I halve the price of my product?”
“Which clients will pay their debts only if I call them?”
>

If done right, this method can yield a more consistent pridiction capability when compared to correlation dependent modeling. 

## How to Use
This isn't an installable package. You can explore the notebooks I used for experimention. These include EDA, Feature Selection, Causal-Graphs, and Causal-Inference. I have also written wrappers for the libraries I used like the causalnex libarry. 


## Tasks Performed

- [x] Perform EDA on the Data
- [x] Perform Feature Selection study
- [x] Split the data in to a training and a holdout set
- [x] Construct a stable causal-graph from the training dataset
  - [x] Create a baseline causal-graph with all the training data
  - [x] Create a causal-graph using 40% of the training data
    - [x] Evaluate their similarity by calculating IoU with Jaccard Index
  - [x] Repeat the above with 70% of the training data
- [x] Ones I get a stable causal-graph, I will chose only the features with a direct connection to the target column. Then do the previous step until I get a stable causal-graph with the new data. 
- [x] I train a Bayesian Network with the two versions of graph and data.
- [x] Evaluate the trained models.

## To Do

- [ ] Build an inference pipeline
- [ ] Making the CML reports more dynamic
- [ ] Adding MLFlow to the causal-graph and Bayesian Network training steps
- [ ] Adding proper doc-strings to all scripts
- [ ] Adding a feature store for the final selected best columns
- [ ] do-calculus?

## Contributors

![Contributors list](https://contrib.rocks/image?repo=Hen0k/causal-inference)

Made with [contrib.rocks](https://contrib.rocks).
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Hen0k/causal-inference.svg?style=for-the-badge
[contributors-url]: https://github.com/Hen0k/causal-inference/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Hen0k/causal-inference.svg?style=for-the-badge
[forks-url]: https://github.com/Hen0k/causal-inference/network/members
[stars-shield]: https://img.shields.io/github/stars/Hen0k/causal-inference.svg?style=for-the-badge
[stars-url]: https://github.com/Hen0k/causal-inference/stargazers
[issues-shield]: https://img.shields.io/github/issues/Hen0k/causal-inference.svg?style=for-the-badge
[issues-url]: https://github.com/Hen0k/causal-inference/issues
[license-shield]: https://img.shields.io/github/license/Hen0k/causal-inference.svg?style=for-the-badge
[license-url]: https://github.com/Hen0k/causal-inference/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/henok-tilaye-b18840151/
[readme-image]: ./notebooks/reports/causal_graph.png