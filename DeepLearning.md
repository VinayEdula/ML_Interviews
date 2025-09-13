1. What is deep learning and how does it differ from traditional machine learning?	
Deep learning uses neural networks with multiple layers to automatically learn features from raw data, while traditional ML requires manual feature engineering. Deep learning can handle unstructured data like images and text more effectively.

2. What is a neural network?
A computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers that process information through weighted connections and activation functions.

3. What is a Multi-layer Perceptron (MLP)?
A feedforward neural network with multiple layers including input, hidden, and output layers. Each neuron uses a nonlinear activation function, enabling it to solve non-linearly separable problems.

4. What are the different types of neural networks?
Main types include: Feedforward Networks (MLP), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN/LSTM/GRU), Generative Adversarial Networks (GAN), and Transformer networks.

5. What is the universal approximation theorem?
States that a neural network with a single hidden layer and finite number of neurons can approximate any continuous function to arbitrary accuracy, given sufficient neurons and appropriate activation functions.

6. What are activation functions and why are they important?
Functions that introduce non-linearity to neural networks, enabling them to learn complex patterns. Common types: ReLU, Sigmoid, Tanh, Softmax. Without them, networks would be linear regardless of depth.

7. What is the difference between AI, Machine Learning, and Deep Learning?
AI is the broader field of making machines intelligent. ML is a subset of AI using algorithms to learn patterns from data. Deep Learning is a subset of ML using neural networks with multiple layers.

8. What is backpropagation and how does it work?
Algorithm for training neural networks by propagating error backwards through the network to update weights. Uses chain rule of calculus to compute gradients and minimize loss function through gradient descent.

9. What is gradient descent?
Optimization algorithm that iteratively adjusts model parameters to minimize a loss function by moving in the direction of steepest descent (negative gradient).

10. What is the learning rate and what happens if it's too high or too low?
Hyperparameter controlling step size in gradient descent. Too high: model may overshoot optimal solution and not converge. Too low: slow convergence and may get stuck in local minima.

11. What is the difference between batch, mini-batch, and stochastic gradient descent?	
Batch GD: uses entire dataset per update. Stochastic GD: uses one sample per update. Mini-batch GD: uses small subset of data per update, balancing efficiency and stability.

12. What are epochs, iterations, and batch size?
Epoch: one complete pass through entire dataset. Iteration: one parameter update step. Batch size: number of samples processed before updating parameters. Iterations per epoch = dataset_size / batch_size.

13. What is the vanishing gradient problem?
Issue where gradients become exponentially smaller in earlier layers during backpropagation, making deep networks difficult to train. Caused by repeated multiplication of small gradients.

14. What is the exploding gradient problem?	
Opposite of vanishing gradients, where gradients become exponentially larger, causing unstable training and parameter updates that are too large.

15. What is overfitting and how can you prevent it?	
When model performs well on training data but poorly on new data. Prevention: regularization (L1/L2), dropout, early stopping, more training data, data augmentation, batch normalization.

16. What is underfitting?	
When model is too simple to capture underlying patterns in data, resulting in poor performance on both training and test data. Solutions: increase model complexity, reduce regularization, train longer.

17. What is the bias-variance tradeoff in deep learning?	

18. What is transfer learning?
Technique of using a pre-trained model (trained on large dataset) as starting point for a new but related task, leveraging learned features to reduce training time and improve performance.

19. What is fine-tuning?
Process of adapting a pre-trained model to a new task by continuing training on new data, typically with lower learning rates and freezing some layers initially.

20. What is end-to-end learning?
Training approach where a model learns to map directly from raw input to final output without intermediate manual feature engineering or pipeline stages.

21. What is the Adam optimizer and how does it work?
Adaptive learning rate optimizer combining momentum and RMSprop. Maintains running averages of gradients and their squared values to adapt learning rates per parameter, often converging faster than SGD.

23. What is the difference between SGD, Adam, and RMSprop optimizers?
SGD: basic gradient descent with optional momentum. RMSprop: adapts learning rate using moving average of squared gradients. Adam: combines momentum and RMSprop with bias correction.

25. What is momentum in gradient descent?
Technique that helps accelerate gradients in relevant direction and dampens oscillations by adding fraction of previous update vector to current update.

28. What is weight initialization and why is it important?
Setting initial values of neural network weights before training. Important because poor initialization can lead to vanishing/exploding gradients, slow convergence, or training failure.

29. What is Xavier/Glorot initialization?
Weight initialization method that sets weights to maintain similar variance across layers, helping prevent vanishing/exploding gradients. Weights drawn from distribution with variance 1/n_inputs.

30. What is batch normalization and why is it used?
Technique normalizing layer inputs to have zero mean and unit variance per batch, reducing internal covariate shift, enabling higher learning rates and faster training.

31. What is layer normalization?
Similar to batch normalization but normalizes across features rather than batch dimension, useful for RNNs and when batch size is small.
What is dropout and how does it prevent overfitting?	Regularization technique randomly setting fraction of neurons to zero during training, forcing network to not rely on specific neurons and improving generalization.
What is gradient clipping?	Technique to prevent exploding gradients by limiting gradient values to maximum threshold, either by scaling or clipping individual gradient components.
What is learning rate scheduling?	Technique of changing learning rate during training, typically decreasing over time to help model converge to better solution. Methods: step decay, exponential decay, cosine annealing.
What is early stopping?	Regularization technique that stops training when model performance on validation set starts degrading, preventing overfitting and reducing training time.
What are loss functions and which ones are commonly used?	Functions measuring difference between predicted and actual values. Common types: MSE (regression), Cross-entropy (classification), Hinge loss (SVM), Huber loss (robust regression).
What is cross-entropy loss?	Loss function commonly used for classification problems, measuring difference between predicted and true probability distributions. Penalizes confident wrong predictions more heavily.
What is the difference between L1 and L2 regularization?	L1: adds sum of absolute weights to loss, promotes sparsity. L2: adds sum of squared weights to loss, prevents large weights. L1 can zero out features, L2 shrinks all weights.
How do you handle class imbalance in deep learning?	Techniques: weighted loss functions, oversampling minority class, undersampling majority class, synthetic data generation (SMOTE), focal loss, ensemble methods.


