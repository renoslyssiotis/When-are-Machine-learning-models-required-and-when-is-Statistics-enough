# When are Machine learning models required and when is Statistics enough
Many of the models that are fitted to data to "learn" any underlying patterns are based on both statistical and ML techniques,
with the latter often being more favoured due to their potential in achieving highly accurate predictions. **A big questions 
that arises**, however, is: Are ML models indeed better than the statistical models? And if they are, which ML model achieves
the highest performance on a dataset? Is there an underlying dataset feature (i.e. meta-feature) that makes a model better 
than another for a specific dataset?

The project aim is to explore whether the meta-features of a dataset can indicate which model, ML or statistical, 
generates the most accurate results on binary classification problems, without running all, or ideally, any of the models. The 
following methods and techniques are implemented in order to be able to answer these questions:
- Success rate ratios (SRR) ranking method
- Regression and multi-linear regression of the performance gain and meta-features
- Symbolic meta-modelling
- KNN meta-learner with average ranks
- Two-level clustering of meta-dataset

## Getting started

### Prerequisites and installing
The required dependencies work with python 3.7. Install the following packages: 
- pandas
- scikit-learn
- scipy
- matplotlib
- pathlib, os, sys
```
conda install sklearn
conda install pandas
conda install scipy
conda install matplotlib
conda install pathlib
```

For symbolic meta-modelling
- mpmath
- sympy
```
conda install mpmath
conda install sympy
```



## Running the KNN meta-learning tests
- Place your test dataset in the **test** folder and name it **test** (e.g. test.csv if a CSV file)
- Open the test.py file 
- Run the test.py to obtain predictions about the best-performing models on your test dataset

## Authors
**Renos Lyssiotis** University of Cambridge, Trinity College<br/>
This work is the code written for the Master thesis of the MEng in Information and Computing Engineering degree

## References
1. [Extended Data Characteristics](https://pdfs.semanticscholar.org/445d/49d1e2c7138943377b3bb834fa775b434258.pdf?_ga=2.206274643.1341933229.1578216039-679275796.1578216039)
2. [Datasets Meta-Feature Desscription for Recommending Feature Selection Algorithm](https://www.fruct.org/publications/ainl-fruct/files/Fil.pdf)
3. [A Comparison of Ranking Methods for Classification Algorithm Selection](https://sci2s.ugr.es/keel/pdf/specific/congreso/brazdil00comparison.pdf)
4. [Ranking Learning Algorithms: Using IBL and
Meta-Learning on Accuracy and Time Results](https://link.springer.com/content/pdf/10.1023/A:1021713901879.pdf)
5. [Meta-Learning and the Full Model
Selection Problem](http://quansun.com/pubs/metalearning_qs_thesis.pdf)
