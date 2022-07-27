## Sentiment Classification on RTMovies sst-5. 

#### Introduction:

Imagine we have to fine tune over 40 different variations of BERT-like language models. We want to do so in a reproductible way, so that we can go to any one of these models, with a simple click share it and it would run a precise copy of it on another machine. That is, replicate the environment, replicate the variables and replicate the data configuration. On top of this, we'd like to run experiments on these models, try diferent things, change the way we feed data to these neural networks. What if I told you we can do all of this, using Google Colab Pro GPU instances which is $9.99/mo.

On the MLOps side: we're going to need a data versioning tool, an experiment tracker and an orchestration tool. 
On the deep learning side, we're going to need a library that incorporates a zoo of these transformer models with a common interface so that we build a small neural network on top of it. We're going to need a library to allow us to easily train these models on GPU, a library that is hackable and would allow us to try unusual things, that has an extensive list of callbacks.

How do we go about building this? Let me tell you what I've done!

For MLOps, I used **ClearML**. It's an end-to-end free open source solution that incorprates everything you need on the MLOps side of the machine learning lifecycle, from data-versioning to experiment tracking to orchestration to pipelines to hyperparameter optimization to deployment. All done in an intutive fashion with a machine learning first approach (unlike say, Airflow) with an amazing dashboard and GUI to boot!

For the deep learning side, I used **HuggingFace** for its amazing zoo of transformers and easy to use interfaces. I also used **Pytorch Lightning** for training the neural networks with its vast library of state of the art technique implementations. It's a wrapper on top of Pytorch, which means at any moment if you need more flexibility you can easily go down to the Pytorch level. Unlike bare Pytorch though, it guides you into best practices, saves you the writing of a lot of boilerplate code which prevents bugs, and overall has a battery of amazing quality of life features such as profiling, quick test run, automatic multi-gpu and DDP training easy to use logging and a more extensive callback library than Keras.
