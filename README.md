# CustomNeuralNetwork
Pythonic implementation of a Neural Network using Gradient descent 

---

# Functionality
A Neural Network with full layer to layer connectivity will be trained based on its input data, using delta based gradient descent for weight modifications. It uses sigmoid activation and no further bias. By default, the data being trained on is a csv from https://archive.ics.uci.edu/ml/datasets/seeds.

---

# Inspiration
The [NN-from-scratch](https://github.com/ankonzoid/NN-from-scratch) library by [ankonzoid](https://github.com/ankonzoid) was my inspiration for this library. It helped me grasp how a gradient descent Neural Network may be implemented. The code for pre-processing the data to be trained on is also heavily inspired by his implementation.

---

# Usage
In the CustomNeuralNetwork.py file, in the main function, various values can be edited to get different result:

| Variable Name | Explanation | Example data |
| ------------- | ----------- | ------------ |
| filename      | The filename of a CSV with training data to be loaded | "seeds_dataset.csv" |
| n_hidden_layers | The layout of the hidden layers inbetween the input layer and the output layer. | [5, 3, 5] # For a layer of 5 nodes followed by a layer of 3 nodes followed by a layer of 5 nodes. |
| n_outputs     | The amount of nodes making up the output of the neural network | 3 |
| l_rate        | The learning rate of the Neural Network. This is a factor in the weight updating | 0.5 |
| n_epochs      | The number of times to pass through each training set | 1000 | 
| n_folds       | In how many parts the training data should be split up. n_folds = 4 would mean that 3/4 of the data is training data, while 1/4 of the data is testing data, and which 25% of the data is testing data changes 4 times. | 4 |

With these values filled, the Neural Network can be run with a simple `python CustomNeuralNetwork.py`.

---

# Output
<pre>
Training and cross-validating...
 Fold 1/4: train acc = 95.57%, test acc = 96.15% (n_train = 158, n_test = 52)
 Fold 2/4: train acc = 94.94%, test acc = 96.15% (n_train = 158, n_test = 52)
 Fold 3/4: train acc = 99.37%, test acc = 96.15% (n_train = 158, n_test = 52)
 Fold 4/4: train acc = 96.84%, test acc = 94.23% (n_train = 158, n_test = 52)
Learning Rate: 0.5 with 1000 Epochs after 35.49s.
Epochs: 1000
<b>Avg train acc = 96.68%</b>
<b>Avg test acc = 95.67%</b>
</pre>

---

# Requirements
* Python 3+
* numpy

---

# Note
This was a little hobby project of mine, and although I did optimize it where possible, I would not recommend using this over the highly optimized keras library. However, if you're as interested as I am in Neural Networks, I would recommend this library (or [NN-from-scratch](https://github.com/ankonzoid/NN-from-scratch)) as a way to learn how such a Neural Network can be implemented.
