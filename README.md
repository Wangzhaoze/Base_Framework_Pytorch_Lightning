# Framework of Neural Network


## 1. Introduction

This is a deep learning code framework based on [Pytorch Lightning](https://lightning.ai/) and [hydra](https://hydra.cc/). All the basic classes are implemented in ```base```, including ```Dataset```, ```Dataloader``` and other common structures under the Pytorch framework, as well as Pytorch Lightning-specific neural network training tools such as ```Dtatamodule```, ```Trainer``` and ```CallBack```. You only need to build each part of the process according to your own needs, and inherit the corresponding nodes in the class to realize the scientific management and monitoring of the entire training process. Set the parameters of the neural network in the configuration file, and hydra will parse the parameters in the yaml file and configure the neural network. In addition, it also provides some commonly used callback methods, which can be assistantly in real-time monitoring of various data of the neural network on the Tensorboard, such as Losses, Metrics, prediction and classification results, etc.

Finally, some demonstrations and usage instructions will be provided, such as using this architecture to implement a neural network for downstream tasks such as classification, semantic segmentation, and self-supervised learning on common classic datasets

- ####  [Pytorch Lightning](https://lightning.ai/)

PyTorch Lightning is an open-source lightweight framework for deep learning research, built on top of PyTorch, designed to simplify the development of training and research processes, and improve code readability and maintainability. PyTorch Lightning provides a set of high-level abstractions that help you organize, train, and evaluate deep learning models more easily.

Key features include:

1. **Simplify the training cycle:** PyTorch Lightning abstracts the details of the training cycle, and handles the training, verification and testing steps through a standardized training cycle, which reduces the writing of repetitive code and improves the readability of the code .

2. **Automatic optimizer selection:** PyTorch Lightning can automatically select the appropriate optimizer according to your model and task, reducing the workload of tuning parameters.

3. **Automatic distributed training:** The framework has built-in distributed training support, which can easily train on multiple GPUs or multiple machines.

4. **Automatic adjustment of learning rate:** PyTorch Lightning supports multiple learning rate schedulers, which can automatically adjust the learning rate to optimize the performance of the model.

5. **TensorBoard integration:** PyTorch Lightning integrates TensorBoard to easily visualize training and validation metrics.

6. **Training process recovery:** When an interruption occurs during training, PyTorch Lightning can help you resume training and continue from where it left off.

7. **Modular design:** PyTorch Lightning encourages modular design of models, data loading, optimizers and other parts to make the code more reusable.

8. **Less boilerplate code:** One of the design goals of PyTorch Lightning is to reduce boilerplate code so that you can focus more on models and research.



- #### [hydra](https://hydra.cc/)

Hydra is an open source software package for configuration management and optimization of application parameters. Its main goal is to help you organize and manage complex configurations more easily, making your code more scalable, reusable, and maintainable. Hydra is widely used in deep learning, machine learning, scientific computing, and other fields that require flexible configuration.

Here are some of the main features and uses of Hydra:

1. **Multi-level configuration:** Hydra allows you to define configurations on multiple levels, from global settings to task-specific settings, so you can better organize your configuration.

2. **Group configuration:** You can use Hydra to group different configuration parameters to better organize and manage configuration options.

3. **Multiple configuration sources:** Hydra supports multiple configuration sources, including command lines, configuration files, environment variables, etc. This makes it more convenient to easily switch configurations in different environments.

4. **Override and merge:** You can override parameters in different levels of configuration, and even override configuration through the command line. Hydra will automatically merge these configurations.

5. **Plugin System:** Hydra provides a plugin system that allows you to define and share custom configurations in your project.

6. **Automatic type conversion:** Hydra can automatically convert parameters to the appropriate type according to the purpose of the configuration parameter.

7. **Application parameter optimization:** With Hydra, you can conduct parameter optimization experiments without modifying the code, which is very useful for model tuning.

8. **Easy integration:** Hydra can be seamlessly integrated with other libraries and frameworks (such as PyTorch, TensorFlow, etc.).


## 2. Contributes

- base
- callbacks
- backbones
- cv_tools
- demo and usage

## 3. Structure

    |- base
      |- base_dataset.py
      |- base_dataloader.py
      |- base_datamodule.py
    |
    |
    |
    |
    |
    |
    |
    |

## 4. Usage