# Build-a-Basic-neural-network

**Aim:** 

Build a Basic neural network from scratch using a high-level library like Tensorflow or Pytorch. Use appropriate dataset.

**Dataset:**
Shoe-Size

**Theory:**

Neural Networks are computational models that mimic the complex functions of the human brain. The neural networks consist of interconnected nodes or neurons that process and learn from data, enabling tasks such as pattern recognition and decision making in machine learning.
Working on a Neural Network
Neural networks are complex systems that mimic some features of the functioning of the human brain. It is composed of an input layer, one or more hidden layers, and an output layer made up of layers of artificial neurons that are coupled. The two stages of the basic process are called backpropagation and forward propagation.

![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/fc6cb9a1-9b6b-4f81-9098-0db28c195cf9)


Forward Propagation

•	Input Layer: Each feature in the input layer is represented by a node on the network, which receives input data.

•	Weights and Connections: The weight of each neuronal connection indicates how strong the connection is. Throughout the training, these weights are changed.

•	Hidden Layers: Each hidden layer neuron processes inputs by multiplying them by weights, adding them up, and then passing them through an activation function. By doing this, non-linearity is introduced, enabling the network to recognize intricate patterns.

•	Output: The final result is produced by repeating the process until the output layer is reached.
Backpropagation

•	Loss Calculation: The network’s output is evaluated against the real goal values, and a loss function is used to compute the difference. For a regression problem, the Mean Squared Error (MSE) is commonly used as the cost function.

Loss Function: 
•	Gradient Descent: Gradient descent is then used by the network to reduce the loss. To lower the inaccuracy, weights are changed based on the derivative of the loss concerning each weight.

•	Adjusting weights: The weights are adjusted at each connection by applying this iterative process, or backpropagation, backward across the network.

•	Training: During training with different data samples, the entire process of forward propagation, loss calculation, and backpropagation is done iteratively, enabling the network to adapt and learn patterns from the data.

•	Activation Functions: Model non-linearity is introduced by activation functions like the rectified linear unit (ReLU) or sigmoid. Their decision on whether to “fire” a neuron is based on the whole weighted input.
Predicting a numerical value

E.g. predicting the price of a product

The final layer of the neural network will have one neuron and the value it returns is a continuous numerical value. To understand the accuracy of the prediction, it is compared with the true value which is also a continuous number.
 
 ![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/f027d5f0-b948-480b-af8f-b6bef5a9f3f4)

Final Activation Function
Linear — This results in a numerical value that we require

![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/c908ce80-510a-4bf1-a01b-709e566cf610)

 
Or ReLU — This results in a numerical value greater than 0
 
 ![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/5b5f7c6e-d0e1-4c69-a409-242e2a906c84)

Loss Function
Mean squared error (MSE) — This finds the average squared difference between the predicted value and the true value
 
 ![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/b246b0e3-ff55-4ada-8178-97710a7bfdc6)

Categorical: Predicting a binary outcome
E.g. predicting a transaction is fraud or not
The final layer of the neural network will have one neuron and will return a value between 0 and 1, which can be inferred as probable.
To understand the accuracy of the prediction, it is compared with the true value. If the data is that class, the true value is a 1, else it is a 0.
 
 ![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/db87c1d6-7bf4-463e-982d-ff01f55625b4)

Final Activation Function
Sigmoid — This results in a value between 0 and 1 which we can infer to be how confident the model is of the example being in the class
 
 ![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/176c8bfa-ce05-49a2-88ab-431868c86a0e)

Loss Function
Binary Cross Entropy — Cross entropy quantifies the difference between two probability distributions. Our model predicts a model distribution of {p, 1-p} as we have a binary distribution. We use binary cross-entropy to compare this with the true distribution {y, 1-y}
 
 ![image](https://github.com/AdityaPatil0718/Build-a-Basic-neural-network/assets/128233555/6d1a4446-17a2-4a8b-b557-79efc2194da7)



**Conclusion:**

Basic neural networks with TensorFlow are effective for simple tasks like the Iris dataset but may need deeper architectures for complex data. Preprocessing, like feature scaling, is crucial. Further experimentation is often necessary for optimal performance on tougher problems. 	


