Batch Normalisation Explained
A detailed guide to how batch normalisation works and the issues it addresses
In this article, I take a detailed look at Batch Normalisation and how it works. Batch Normalisation was introduced in 2015 by Loffe and Szegedy and quickly became a standard feature implemented in almost every deep network.

Outline
Internal Covariate Shift
Vanishing and exploding gradients
How does Batch Normalisation work?
Advantages of Batch Normalisation
1. Internal Covariate Shift
The key issue that batch normalisation tackles is internal covariate shift. Internal covariate shift occurs due to the very nature of neural networks. At every epoch of training, weights are updated and different data is being processed, which means that the inputs to a neuron is slightly different every time. As these changes get passed on to the next neuron, it creates a situation where the input distribution of every neuron is different at every epoch.

Normally, this is not a big deal, but in deep networks, these small changes in input distribution add up fast and amplify greatly deeper into the network. Ultimately, the input distribution received by the deepest neurons changes greatly between every epoch.

As a result, these neurons need to continuously adapt to the changing input distribution, meaning that their learning capabilities are severely bottlenecked. This constantly changing input distribution is called internal covariate shift.

2. Vanishing and exploding gradients
Another issue that batch normalisation tackles is vanishing or exploding gradients. Before Rectified Linear Units (ReLUs), saturated activation functions were used. A saturated function is one that has a “flattened” curve towards to the left and right bounds, such as the sigmoid function.


Sigmoid curve and its derivative
In the sigmoid function, the gradient tends towards 0 as the value of x tends towards ±∞. As a neural network is trained, the weights can be pushed towards the saturated ends of the sigmoid curve. As such, the gradient gets smaller and smaller and approaches 0.

These small gradients get even smaller when multiplied together deeper into the network. When using backpropagation, the gradient gets exponentially closer to 0. This “vanishing” gradient severely limits the depth of networks.

Although this vanishing gradient can be easily managed by using a non-saturated activation function such as ReLU, batch normalisation still has a place as it prevents the weights from being pushed to those saturated regions in the first place, by ensuring no value has gone too high or low.

3. How does Batch Normalisation work?
Batch normalisation normalises a layer input by subtracting the mini-batch mean and dividing it by the mini-batch standard deviation. Mini-batch refers to one batch of data supplied for any given epoch, a subset of the whole training data.


Formula for batch normalisation, where x̂ refers to the normalised vector.
The normalisation ensures that the inputs have a mean of 0 and a standard deviation of 1, meaning that the input distribution to every neuron will be the same, thereby fixing the problem of internal covariate shift and providing regularisation.


However, the representational power of the network has been severely compromised. If each layer is normalised, the weight changes made by the previous layer and noise between data is partially lost, as some non-linear relationships are lost during normalisation. This can lead to suboptimal weights being passed on.

To fix this, batch normalisation adds two trainable parameters, gamma γ and beta β, which can scale and shift the normalised value.


Stochastic gradient descent can tune γ and β during standard backpropagation to find the optimal distribution such that the noise between data and sparseness of the weight changes are accounted for. Essentially, these parameters scale and shift the normalised input distribution to suit the peculiarities of the given dataset.

For example, given that an un-normalised input distribution is best for a given dataset, γ and β will converge to √Var[x] and E[x], such that the original un-normalised x vector is obtained. Hence, batch normalisation ensures that the normalisation is always optimal for the given dataset.

a. Why normalise with respect to a mini-batch?
Ideally, the normalisation should be with respect to the entire training data set, as this ensures there will be no change in input distribution between different batches. However, since any dataset not in the current batch is outside the scope of backpropagation, stochastic gradient descent would not work, since the statistics used in the normalisation comes from outside the scope.

Hence, the normalisation is done with respect to a mini-batch to ensure that standard backpropagation can be done. The only implication is that each batch should be somewhat representative of the distributions of the entire training set, which is a safe assumption if your batch size is not too small.

b. Test phase
During training, the mean and standard deviation are calculated using samples in the mini-batch. However, in testing, it does not make sense to calculate new values. Hence, batch normalisation uses a running mean and running variance that is calculated during training. There is a need to introduce a new parameter, momentum or decay.

running_mean = momentum * running_mean + (1-momentum) * new_mean
running_var = momentum* running_var + (1-momentum) * new_var
Momentum is the importance given to the last seen mini-batch, a.k.a “lag”. If the momentum is set to 0, the running mean and variance come from the last seen mini-batch. However, this may be biased and not the desirable one for testing. Conversely, if momentum is set to 1, it uses the running mean and variance from the first mini-batch. Essentially, momentum controls how much each new mini-batch contributes to the running averages.

Ideally, the momentum should be set close to 1 (>0.9) to ensure slow learning of the running mean and variance such that the noise in a mini-batch is ignored.

4. Advantages of Batch Normalisation
a. Larger learning rates
Typically, larger learning rates can cause vanishing/exploding gradients. However, since batch normalisation takes care of that, larger learning rates can be used without worry.

b. Reduces overfitting
Batch normalisation has a regularising effect since it adds noise to the inputs of every layer. This discourages overfitting since the model no longer produces deterministic values for a given training example alone.

Conclusion
The power of Batch Normalisation has been repeatedly shown in many areas of Machine Learning. It is a simple drop in solution that will yield a significant improvement in performance for almost any model.
