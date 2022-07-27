## Sentiment Classification on RTMovies sst-5. 

#### Introduction:

Imagine we have to fine-tune over 40 different variations of BERT-like language models. We want to do so in a reproducible way, so that we can go to any one of these models, with a simple click share it and it would run a precise copy of it on another machine. That is, replicate the environment, replicate the variables and replicate the data configuration. On top of this, we'd like to run experiments on these models, try different things, and change the way we feed data to these neural networks. We'd also like to do this on Google Cola Pro GPU instances. 

On the MLOps side: we're going to need a data versioning tool, an experiment tracker, and an orchestration tool. 
On the deep learning side, we're going to need a library that incorporates a zoo of these transformer models with a common interface so that we build a small neural network on top of it. We're going to need a library to allow us to easily train these models on GPU, a library that is hackable and would allow us to try unusual things, and that has an extensive list of callbacks.

How do we go about building this? Let me tell you what I've done!

For MLOps, I used **ClearML**. It's an end-to-end free open source solution that incorporates everything you need on the MLOps side of the machine learning lifecycle, from data-versioning to experiment tracking to orchestration to pipelines to hyperparameter optimization to deployment. All done intuitively with a machine learning first approach (unlike say, Airflow) with an amazing dashboard and GUI to boot!

On the deep learning side, I used **HuggingFace** for its zoo of transformers and easy-to-use interfaces. I also used **Pytorch Lightning** for training the neural networks with its vast library of the state of the art technique implementations. It's a wrapper on top of Pytorch, which means at any moment if you need more flexibility you can easily go down to the Pytorch level. Unlike bare Pytorch though, it guides you into best practices and saves you the writing of boilerplate code which helps prevent bugs. Overall, the library has a battery of quality-of-life features such as profiling, quick test run, automatic multi-GPU, and DDP training, easy-to-use logging, and useful callbacks for learning rate monitoring, model checkpointing, early stopping, network pruning, and many more.

#### The Data:
The data is the Rotten Tomatoes movie reviews sentiment tree-bank dataset. It has 5 classes:
 - 0 : Negative.
 - 1 : Slightly negative.
 - 2 : Neutral.
 - 3 : Slightly positive.
 - 4 : Positive.

Here is a wordcloud of the most frequent positive words:
![plot](./img/sentiment_4.png)

Being a sentiment tree-bank means the reviews are decomposed into sequences of their constituent n-grams, which were then labeled separately. This implies there are a lot of single words (1-grams) in the data, the majority of which have neutral values, we need to account for this imbalance.

We can see this in the overall class distribution plot, where about half of the examples fall in the neutral category:
![plot](./img/class_distribution.png)

We can even see how the class distribution changes as the n-grams increase in size:
![plot](./img/n_gram_dist.svg)

As expected the larger the text, the more uniform the distribution becomes among the classes.

##### Baselines:
We can establish some quick classification baselines. A purely random classifier would achieve:
$$P(y = k) = \sum_{j = 0}^{4} P(j)P(y = j) \approx 0.19 \\ \text{accuracy.}$$
Whereas a majority class classifier would achieve:
$$P(y = k) = P(k) \approx 0.50 \\ \text{accuracy}$$

##### Train/Val/Test splits:
To prevent target leakage we have to split the data in such a way that we preserve the entire tree for each review. That is, for each review either all of the tree is on the training set or the validation set, or the test set, where the pairwise intersection of these sets is empty. In other words, we cannot have nodes corresponding to a review in the test set appear in the training set.

Arbitrarily splitting the data, as the Kaggle competition did, would not result in a model that can correctly generalize.
The first step is to slice away the testing set, this is done in **test_data_split.py**. The data is retrieved from an S3 bucket and placed into a ClearML Dataset task.
The split is done with the consideration above, and the seed is chosen such that the Wasserstein distance between the resulting distributions is minimized, to ensure that the testing data is representative of the training data.

