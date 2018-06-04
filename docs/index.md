# Introduction to Machine Learning

!!! note
    The content of this website was automatically generated from Jupyter notebooks.
    
    Hopefully, nothing has been broken.

Lectures notes are prepared in [Colaboratory](https://colab.research.google.com/), which is a Jupyter notebook environment running in the cloud and storing notebooks in Google Drive. It makes it easy to collaborate on a project in Jupyter notebooks and provides free computing power.

This website was created for your convenience. Feel free to use Jupyter notebooks though:

* Introduction ([Colaboratory](https://colab.research.google.com/drive/1qJj4jZMpBpfCkHc0bavFGezx8bhJlVcx) or [GitHub](https://github.com/TomaszGolan/introduction_to_machine_learning/blob/master/docs/notebooks/introduction_to_machine_learning_00_intro.ipynb))

* k-Nearest Neighbors ([Colaboratory](https://colab.research.google.com/drive/1My8UggN12Opt_gscK3tl4VLhZkHiQSyX) or [GitHub](https://github.com/TomaszGolan/introduction_to_machine_learning/blob/master/docs/notebooks/introduction_to_machine_learning_01_knn.ipynb))

    * [First ML problem](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#our-first-ml-problem)

    * [Nearest Neighbor algorithm](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#nearest-neighbor)
    
    * [k-Nearest Neighbors algorithm](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#k-nearest-neighbors_1)

    * [Hyperparameters](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#hyperparameters)

    * [Iris dataset](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#iris-dataset)

    * [MNIST](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#mnist)

    * [Regression with kNN](markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#regression-with-knn)

    * [Summary](/markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/#summary)

* Decision trees ([Colaboratory](https://colab.research.google.com/drive/1_Qb92Hj5_f2rpta67JC0JKXwE2581Ar-) or [GitHub](https://github.com/TomaszGolan/introduction_to_machine_learning/blob/master/docs/notebooks/introduction_to_machine_learning_02_dt.ipynb))

    * [Introduction](markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#introduction)

    * [ID3 and C4.5 algorithms](markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#id3-and-c45-algorithms)

    * [CART](markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#cart)

    * [Bias-variance trade-off](markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#bias-variance-trade-off)

    * [Ensemble learning](markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#ensemble-learning)

    * [Summary](markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#summary_2)

* Support vector machine ([Colaboratory](https://colab.research.google.com/drive/1IA_RgU64I8OZ-KKNV42T4ldkEOHFZ8d_) or [GitHub](https://github.com/TomaszGolan/introduction_to_machine_learning/blob/master/docs/notebooks/introduction_to_machine_learning_03_svm.ipynb))

    * [Linear SVM](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#linear-svm)
    
    * [Lagrange multipliers](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#lagrange-multipliers)
    
    * [Optimal margin](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#optimal-margin)
    
    * [Non-linear SVM](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#non-linear-svm)
    
    * [Soft margin](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#soft-margin)
    
    * [SMO algorithm](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#smo-algorithm)
    
    * [Examples](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#examples)
    
    * [Multiclass classification](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#multiclass-classification)
    
    * [SVM regression](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#svm-regression)
    
    * [Summary](markdown/introduction_to_machine_learning_03_svm/introduction_to_machine_learning_03_svm/#summary)

* Neural networks ([Colaboratory](https://colab.research.google.com/drive/1DdGmph_WzVpCRJ2c6jVRDcznJ--8xduh) or [GitHub](https://github.com/TomaszGolan/introduction_to_machine_learning/blob/master/docs/notebooks/introduction_to_machine_learning_04_nn.ipynb))

    * [Linear regression](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#linear-regression)

    * [Logistic regression](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#logistic-regression)

    * [Multinominal logistic regression](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#multinominal-logistic-regression)

    * [Neural networks](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#neural-networks)

    * [AND, OR, XOR](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#and-or-vs-xor)

    * [Simple regression with NN](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#simple-regression-with-nn)

    * [MNIST with softmax](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#mnist)

    * [Gradient descent variations](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#gradient-descent-variations)

    * [Regularization](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#regularization)

    * [Summary](markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/#summary)

* Deep learning ([Colaboratory](https://colab.research.google.com/drive/1pW-SvZ62L-WZRtRyZ-rFETt5L24V7XEz) or [GitHub](https://github.com/TomaszGolan/introduction_to_machine_learning/blob/master/docs/notebooks/introduction_to_machine_learning_05_dl.ipynb))

    * [Convolutional Neural Networks](markdown/introduction_to_machine_learning_05_dl/introduction_to_machine_learning_05_dl/#convolutional-neural-networks)
    
    * [Deep MNIST](markdown/introduction_to_machine_learning_05_dl/introduction_to_machine_learning_05_dl/#deep-mnist)
    
    * [Batch normalization](markdown/introduction_to_machine_learning_05_dl/introduction_to_machine_learning_05_dl/#batch-normalization)
    
    * [Data augmentation](markdown/introduction_to_machine_learning_05_dl/introduction_to_machine_learning_05_dl/#data-augmentation)
    
    * [Summary](markdown/introduction_to_machine_learning_05_dl/introduction_to_machine_learning_05_dl/#summary_1)
    

