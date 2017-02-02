---
layout: post
title: Tell the species of Iris flower - Logistic Regression with Tensorflow 
excerpt_separator: <!--more-->
---
![]({{ site.baseurl }}/images/tf_logisticregression_deco.png)
<!--more-->

This post is to model a **logistic regression** between 4 features of Iris (sepal length, sepal width, petal lenght, petal width) and species (setosa, versicolor, virginica). With this model, given the four measures from any new Iris belonging to one of the three species, we could tell which species it belongs to.

In 1936, Ronald Fisher introduced *Fisher's Iris data set* in his paper for taxonomic problems. It is also called *Anderson's Iris data set*, because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.

In general, the logistic model is a variation of Linear Regression, with the observed dependent variable y being categorical. The formula predicts the probability of the class label as a function of the independent variables: p = 1/(1+e^-x^). Here we use the model Y = 1 / (1+e^WX+B^) = sigmoid(W*X+B), where X is the features [150 X 4] and y is the predicted specie [150 X 3]. Y is one-hot vector where it is [1,0,0], [0,1,0], or [0,0,1] for the 3 species. W is the weight for X and B is the bias. The idea is to minimize the difference between the predicted label "y" and the acturual label "Y", in the form of total squared error loss function = sum of (y - Y)^2. With an initial "W" and "B" assigned, they were updated each iteration to minimize the loss with gradient descent optimization. The training is stopped when the change of the loss in two consecutive iterations is smaller than a preset convergence tolerence level (e.g. 0.0008).

**Important**: We did not do data normalization for this dataset since the data are close to 1 and the range is not that big. Actually I have tried to do the regression with the data normalized based on training data's mean and standard deviation, the performance is not as good as with the original data, even though every other parameter was kept the same. So one may need to check the data's range before performing the classification task.


Source code:
    trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42) </div>


67% training data, and 33% testing data.

Cost function:

```{r eval=FALSE}
y = tf.nn.sigmoid(tf.add(tf.matmul(X, weights)))
loss = tf.nn.l2_loss(y-Y, name="squared_error_cost")
```
Optimize through learning:
```{r eval=FALSE}
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```
Learning rate is selected as 0.0008 whose result are shown below (90% accuracy on the testing data set). However, when tuning the learning rate to 0.008, the testing accuracy is 100%. You can try it out yourself to check the loss changes.


![]({{ site.baseurl }}/images/tf_logistic_result.png)

Figure 1 shows the first two features of the samples for the three groups, based on the original data (top panel), and based on the predicted label (bottom panel). 

### Tensorboard

Tensorboaf would help you to visualize the training steps and the change of the variables of interest. 
In general, there are several levels to include:
1. tf.Graph()
2. tf.namescope('names')
3.1 tf.summary.scalar('varaible_name',variable)
3.2 tf.summary.histogram('histogram', variable)
3.3 tf.summary.image("plot", image)
4. merge the summaries: merged=tf.summary.merge_all()    
5. appoint a location to write the summary: train_writer = tf.summary.FileWriter('path_to_save', sess.graph)
6. initialize: tf.global_variables_initializer().run()
7. update the summary for each iteration: train_writer.add_summary(summary, iteration)
8. after the program has finished, launch tensorboard:
tensorboard --logdir=path/to/log-directory
Once TensorBoard is running, navigate your web browser to localhost:6006 to view the TensorBoard.

With tensorboard, we could see the change of "W", "B", and "loss" during training.

![]({{ site.baseurl }}/images/tf_logistic_scalar.png)

The final figure shows the graph of the model from input to the training. 

![]({{ site.baseurl }}/images/tf_logistic_graph.png)


### Information
* Source code could be found on [my github](https://github.com/pestoo0221/tensorflow_logisticregressio).

* Detailed information about the data set could be found [here](https://en.wikipedia.org/wiki/Iris_flower_data_set).