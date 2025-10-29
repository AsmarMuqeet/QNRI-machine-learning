# Qiskit Machine Learning For eX3, Sigma2, and VLQ

[![License](https://img.shields.io/github/license/qiskit-community/qiskit-machine-learning.svg?)](https://opensource.org/licenses/Apache-2.0) <!--- long-description-skip-begin -->
[![Current Release](https://img.shields.io/github/release/qiskit-community/qiskit-machine-learning.svg?logo=Qiskit)](https://github.com/qiskit-community/qiskit-machine-learning/releases)
[![Build Status](https://github.com/qiskit-community/qiskit-machine-learning/actions/workflows/main.yml/badge.svg)](https://github.com/qiskit-community/qiskit-machine-learning/actions?query=workflow%3A"Machine%20Learning%20Unit%20Tests"+branch%3Amain+event%3Apush)
[![Coverage Status](https://coveralls.io/repos/github/qiskit-community/qiskit-machine-learning/badge.svg?branch=main)](https://coveralls.io/github/qiskit-community/qiskit-machine-learning?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qiskit-machine-learning)
[![Monthly downloads](https://img.shields.io/pypi/dm/qiskit-machine-learning.svg)](https://pypi.org/project/qiskit-machine-learning/)
[![Total downloads](https://static.pepy.tech/badge/qiskit-machine-learning)](https://pepy.tech/project/qiskit-machine-learning)
[![Slack Organisation](https://img.shields.io/badge/slack-chat-blueviolet.svg?label=Qiskit%20Slack&logo=slack)](https://slack.qiskit.org)
[![arXiv](https://img.shields.io/badge/arXiv-2505.17756-b31b1b.svg)](https://arxiv.org/abs/2505.17756)

<!--- long-description-skip-end -->

## What is Qiskit Machine Learning?

Qiskit Machine Learning introduces fundamental computational building blocks, such as Quantum 
Kernels and Quantum Neural Networks, used in various applications including classification 
and regression.

This library is a fork of Qiskit machine learning for integration with eX3, Sigma2, and VLQ. The Qiskit machine learning library is part of the Qiskit Community ecosystem, a collection of high-level codes that are based
on the Qiskit software development kit.

> [!NOTE]
> A description of the original library structure, features, and domain-specific applications, can be found 
> in a dedicated [ArXiv paper](https://arxiv.org/abs/2505.17756).

The Qiskit Machine Learning framework aims to be:

* **User-friendly**, allowing users to quickly and easily prototype quantum machine learning models without 
    the need of extensive quantum computing knowledge.
* **Flexible**, providing tools and functionalities to conduct proof-of-concepts and innovative research 
    in quantum machine learning for both beginners and experts.
* **Extensible**, facilitating the integration of new cutting-edge features leveraging Qiskit's 
    architectures, patterns and related services.


## What are the main features of Qiskit Machine Learning?

### Kernel-based methods

The FidelityQuantumKernel class uses the Fidelity algorithm. It computes kernel matrices for datasets and can be combined with a Quantum Support Vector Classifier (QSVC) 
or a Quantum Support Vector Regressor (QSVR) to solve classification or regression problems respectively. It is also compatible with classical kernel-based machine learning algorithms.


### Quantum Neural Networks (QNNs)

Qiskit Machine Learning defines a generic interface for neural networks, implemented by two core (derived) primitives:

- EstimatorQNN, combining parametrized quantum circuits with Z-pauli quantum observables. The output is the expected value of the Z observable.
  
- SamplerQNN, translating bit-string counts into the desired outputs.

To train and use neural networks, Qiskit Machine Learning provides learning algorithms such as the NeuralNetworkClassifier and NeuralNetworkRegressor.
Finally, built on these, the Variational Quantum Classifier (VQC) and the Variational Quantum Regressor (VQR),
take a _feature map_ and an _ansatz_ to construct the underlying QNN automatically using high-level syntax.

### Integration with PyTorch

The [`TorchConnector`](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.connectors.TorchConnector.html#qiskit_machine_learning.connectors.TorchConnector) 
integrates QNNs with [PyTorch](https://pytorch.org). 
Thanks to the gradient algorithms in Qiskit Machine Learning, this includes automatic differentiation. 
The overall gradients computed by PyTorch during the backpropagation take into account quantum neural 
networks, too. The flexible design also allows the building of connectors to other packages in the future.

## Installation and documentation
```bash
pip install qiskit-machine-learning
```

`pip` will install all dependencies automatically, so that you will always have the most recent
stable version.

### Optional Installs

* **PyTorch** may be installed either using command `pip install 'qiskit-machine-learning[torch]'` to install the
  package or refer to PyTorch [getting started](https://pytorch.org/get-started/locally/). When PyTorch
  is installed, the `TorchConnector` facilitates its use of quantum computed networks.

* **Sparse** may be installed using command `pip install 'qiskit-machine-learning[sparse]'` to install the
  package. Sparse being installed will enable the usage of sparse arrays and tensors.

* **NLopt** is required for the global optimizers. [`NLopt`](https://nlopt.readthedocs.io/en/latest/) 
  can be installed manually with `pip install nlopt` on Windows and Linux platforms, or with `brew 
  install nlopt` on MacOS using the Homebrew package manager. For more information, 
  refer to the [installation guide](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/).

----------------------------------------------------------------------------------------------------

### Creating Your First Machine Learning Programming Experiment in Qiskit

Now that Qiskit Machine Learning is installed, it's time to begin working with the Machine 
Learning module. Let's try an experiment using VQC (Variational Quantum Classifier) algorithm to
train and test samples from a data set to see how accurately the test set can be classified.

```python
from qiskit.circuit.library import n_local, zz_feature_map
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

seed = 1376
algorithm_globals.random_seed = seed

# Use ad hoc data set for training and test data
feature_dim = 2  # dimension of each data point
training_size = 20
test_size = 10

# training features, training labels, test features, test labels as np.ndarray,
# one hot encoding for labels
training_features, training_labels, test_features, test_labels = ad_hoc_data(
    training_size=training_size, test_size=test_size, n=feature_dim, gap=0.3
)

feature_map = zz_feature_map(feature_dimension=feature_dim, reps=2, entanglement="linear")
ansatz = n_local(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
)
vqc.fit(training_features, training_labels)

score = vqc.score(test_features, test_labels)
print(f"Testing accuracy: {score:0.2f}")
```

### More examples

Learning path notebooks may be found in the
[Machine Learning tutorials](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/index.html) section
of the documentation and are a great place to start. 

## License

This project uses the [Apache License 2.0](https://github.com/qiskit-community/qiskit-machine-learning/blob/main/LICENSE.txt).
