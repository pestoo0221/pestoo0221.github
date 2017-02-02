---
layout: post
title: Tell temperature through a cricket (biological data) - Linear Regression with Tensorflow 
excerpt_separator: <!--more-->
---
![]({{ site.baseurl }}/images/tf_linearregression_deco.png)

<!--more-->

This post is to model a linear relationship between "chirps of a cricket" and ground temperature using tensorflow. With this model, we could tell the temperature by listening to the songs pf the cricket!

In 1948, Pierce recorded the number of chirps (per 15 second) made by Crickets at different ground temperatures. Because Crickets are "cold-blooded", their physicological process is believed to be affected by temperature. He actually found that there is pattern in the way crickets respond to the change in ground temperature between 60 to 100 degrees of farenhite. It was also found that Crickets did not sing at temperatures above or below the range.

The data is derieved from Pierce's book "The Song of Insects", 1948. We aim to fit a linear model and find the best fit line for the given "Chirps(per 15 Second)" and the corresponding "Temperatures(Farenhite)" using TensorFlow. 

In general, the linear model is y = W * X + B, where X is the number of Chirps and y is the predicted temperature. W is the weight for X and B is the bias, or interception. The idea is to minimize the difference between the predicted temperature "y" and the acturual temperature "Y", in the form of mean squared error loss function = mean of (y - Y)^2^. With an initial "W" and "B" assigned, they were updated each iteration to minimize the loss with gradient descent optimization. The training is stopped when the change of the loss in two consecutive iterations is smaller than a preset convergence tolerence level (e.g. 0.0001).

**Important**: Before we apply the model with the input data, we need to normalize the data to help to improve the performance of the gradient descent. We normalize the data to be with mean of 0 and standard deviation of 1. X_new = (X - mean(X))/std(X). You can try with the data without normalizing them first. In that case, changing the initial value of W and B would affect the final result a lot. 

Source code:
```{r eval=FALSE}
x_data_n, y_data_n = ((x_data - x_data.mean())/x_data.std(),(y_data - y_data.mean())/y_data.std() )
```
Cost function:
```{r eval=FALSE}
loss = tf.reduce_mean(tf.squared_difference(y_predicted,y_data_n))
```

Optimize through learning:
```{r eval=FALSE}
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```
Learning rate is selected in the range of 0.0001 to 10 (boundary included), the smaller the rate is, the more steps it takes to converge; vice versa. However, if it is too small, it will take too long to converge, and if it is too big, it may jump over the convergence point to the other extreme...We chose 0.1 in our model. 


![]({{ site.baseurl }}/images/linearregression_figure_1.png)

Figure 1 shows the normalized data (red dots, top left) with the final best fit line, the change of loss over steps (bottom left), the moving of the fitted line to its best location with steps (top right, green-purple), and the original data with final best fit line.

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

![]({{ site.baseurl }}/images/linearregression_tensorboard_scalar.png)

The "line of fitness" is plotted overlayed with the original data. 
 
![]({{ site.baseurl }}/images/linearregression_tensorboard_image.png)

The final figure shows the graph of the model from input to the training. 

![]({{ site.baseurl }}/images/linearregression_tensorboard_graph.png)

With this model, one could easily tell what temperature it is just by listening to the songs of cricket. 

### Quick question

If the ground temperature should drop to 32º F, what happens to the cricket's chirping rate? (answer is below)
.
.
.
The crickets died.... This is biological data, so one need pay attention to the reasonable range of the data.

### Information
* Source code could be found on [my github](https://github.com/pestoo0221/tesnforflow_lineargression)

* The Song of Insects by George W. Pierce, 1948

* A traditional way to fit the linear regression: please head over to [the website](http://mathbits.com/MathBits/TISection/Statistics2/linearREAL.htm)