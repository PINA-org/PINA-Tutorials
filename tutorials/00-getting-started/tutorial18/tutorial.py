#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Introduction to Solver classes
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINA-org/PINA-tutorials/blob/main/tutorials/tutorial18/tutorial.ipynb)
# 
# In this tutorial, we will explore the Solver classes in PINA, that are the core components for optimizing models. Solvers are designed to manage and execute the optimization process, providing the flexibility to work with various types of neural networks and loss functions. We will show how to use this class to select and implement different solvers, such as Supervised or Physics-Informed solvers. By the end of this tutorial, you'll be equipped to easily choose and customize solvers for your own tasks, streamlining the model training process.
# 
# ## Introduction to Solvers
# 
# [`Solvers`](https://PINA-org.github.io/PINA/_rst/_code.html#solvers) are versatile objects in PINA designed to manage the training and optimization of machine learning models. They handle key components of the learning process, including:
# 
# - Loss function minimization  
# - Model optimization (optimizer, schedulers)
# - Validation and testing workflows
# 
# PINA solvers are built on top of the [PyTorch Lightning `LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), which provides a structured and scalable training framework. This allows solvers to leverage advanced features such as distributed training, early stopping, and logging — all with minimal setup.
# 
# ## Solvers Hierarchy: Single-Model, Multi-Model, and Ensemble
# 
# PINA provides three main base solver classes, designed respectively for training a single model, multiple distinct models, or an ensemble of models. These interfaces define the core functionality inherited by all specific solver implementations.
# 
# ### 1. [`SingleModelSolver`](https://PINA-org.github.io/PINA/_rst/solver/single_model_solver.html)
# 
# This is the base class for solvers that train **a single model**, such as in standard supervised learning or physics-informed training.
# 
# **Arguments:**
# - `problem` – The problem to be solved.
# - `model` – The neural network model.
# - `optimizer` – Defaults to `torch.optim.Adam` if not provided.
# - `scheduler` – Defaults to `torch.optim.lr_scheduler.ConstantLR`.
# - `weighting` – Optional loss weighting schema., see [here](https://PINA-org.github.io/PINA/_rst/_code.html#losses-and-weightings).
# - `loss` - Defaults to `torch.nn.MSELoss`.
# - `use_lt` – Whether to use LabelTensors as input.
# 
# ---
# 
# ### 2. [`MultiModelSolver`](https://PINA-org.github.io/PINA/_rst/solver/multi_model_solver.html)
# 
# This is the base class for solvers involving **multiple models** with distinct behaviours, such as in `SelfAdaptivePhysicsInformedSolver`.
# 
# **Arguments:**
# - `problem` – The problem to be solved.
# - `models` – The models used for training.
# - `optimizers` – Defaults to `torch.optim.Adam`.
# - `schedulers` – Defaults to `torch.optim.lr_scheduler.ConstantLR`.
# - `weighting` – Optional loss weighting schema, see [here](https://PINA-org.github.io/PINA/_rst/_code.html#losses-and-weightings).
# - `loss` - Defaults to `torch.nn.MSELoss`.
# - `use_lt` – Whether to use LabelTensors as input.
# 
# ---
# 
# ### 3. [`EnsembleSolver`](https://PINA-org.github.io/PINA/_rst/solver/ensemble_solver.html)
# 
# This is the base class for solvers that train **an ensemble of models** with the same behaviour, such as in `SupervisedEnsembleSolver`.
# 
# **Arguments:**
# - `problem` – The problem to be solved.
# - `models` – The models used for training.
# - `optimizers` – Defaults to `torch.optim.Adam`.
# - `schedulers` – Defaults to `torch.optim.lr_scheduler.ConstantLR`.
# - `weighting` – Optional loss weighting schema, see [here](https://PINA-org.github.io/PINA/_rst/_code.html#losses-and-weightings).
# - `loss` - Defaults to `torch.nn.MSELoss`.
# - `use_lt` – Whether to use LabelTensors as input.
# 
# ---
# 
# These base classes define the structure and behavior of solvers in PINA, enabling customized training strategies while leveraging PyTorch Lightning under the hood. 
# 
# ## Structure a Solver with Multiple Inheritance:
# 
# Thanks to PINA’s modular design, custom solvers can be easily built through **multiple inheritance**. By combining a chosen base solver class with one or more [`Mixins`](https://PINA-org.github.io/PINA/_rst/_code.html#mixins), users can define flexible and specialized optimization logic. Each mixin encapsulates an isolated component of the solver behavior, making it possible to assemble reusable building blocks into a fully functional custom solver. Examples include:
# 
# - **`PhysicsInformedSingleModelSolver`**
#   - Inherits from:
# 
#     - [`PhysicsInformedMixin`](https://PINA-org.github.io/PINA/_rst/solver/mixin/physics_informed_mixin.html) → enables gradient computation for validation and test phases
#     - [`SingleModelSolver`](https://PINA-org.github.io/PINA/_rst/solver/single_model_solver.html) → manages training of a single model
# 
# - **`AutoregressiveEnsembleSolver`**
#   - Inherits from:  
# 
#     - [`AutoregressiveMixin`](https://PINA-org.github.io/PINA/_rst/solver/mixin/autoregressive_mixin.html) → implements autoregressive logic
#     - [`EnsembleSolver`](https://PINA-org.github.io/PINA/_rst/solver/ensemble_solver.html) → manages training of an ensemble model
# 
# This structure promotes **code reuse** and **extensibility**, allowing you to quickly prototype new solver strategies by reusing core training and optimization logic.
# 
# ## Let's try to build some solvers!
# 
# We will now start building a supervised single-model solver in PINA. Let's first import useful modules! 

# In[ ]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab[tutorial]"')

import warnings
import torch

warnings.filterwarnings("ignore")

from pina import Trainer
from pina.solver import BaseSolver
from pina.solver.mixin import SingleModelMixin, ConditionAggregatorMixin
from pina.condition import InputTargetCondition
from pina.model import FeedForward
from pina.problem.zoo import SupervisedProblem


# We inherit from three classes:
# 
# - `SingleModelMixin`: manages the forward pass logic for a single model
# - `ConditionAggregationMixin`: combines the losses from one or more conditions into a single quantity
# - `BaseSolver`: initializes the solver components

# In[ ]:


class MyFirstSolver(SingleModelMixin, ConditionAggregatorMixin, BaseSolver):

    # Accepted condition types for this solver
    accepted_conditions_types = (InputTargetCondition,)

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        use_lt=True,
    ):
        # Initialize the base solver
        BaseSolver.__init__(self, problem=problem, use_lt=use_lt)

        # Initialize the components of the solver
        self._init_solver_components(
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
        )

        # Initialize the weighting scheme for the conditions and the loss
        self._init_weighting_and_loss(weighting=weighting, loss=loss)


# When a class inherits from multiple parent classes, Python determines the order in which methods are resolved using the method resolution order (MRO). In this case, because `SingleModelMixin` and `ConditionAggregationMixin` appear before `BaseSolver` in the inheritance list, their initialization logic is handled first, before delegating to the `__init__` method of `BaseSolver`. This allows the custom solver to extend or modify the initialization process while still relying on the base functionality provided by `BaseSolver`.

# Let's use `MyFirstSolver` to solve a very simple problem: $y=\sin(x)$.

# In[ ]:


# get the data
x = torch.linspace(0, torch.pi, 100).view(-1, 1)
y = torch.sin(x)
# build the problem
problem = SupervisedProblem(x, y)
# build the model
model = FeedForward(1, 1)


# In[ ]:


# initialize (we use plain tensors!)
solver = MyFirstSolver(problem=problem, model=model, use_lt=False)

# simple training
trainer = Trainer(
    solver, max_epochs=500, train_size=0.8, test_size=0.2, accelerator="cpu"
)
trainer.train()
trainer.test()


# ## A Summary on Solvers
# 
# Solvers in PINA play a critical role in training and optimizing machine learning models, especially when working with complex problems like physics-informed neural networks (PINNs) or standard supervised learning. Here’s a quick recap of the key concepts we've covered:
# 
# 1. **Solver base classes**:
#    - **`SingleModelSolver`**: For solvers using a single model.
#    - **`MultiModelSolver`**: For solvers using multiple models.
#    - **`EnsembleSolver`**: For solvers using an ensemble of models.
# 
# 3. **Custom Solver Implementation**:
#    - Custom solvers can be created by inheriting from one of PINA’s base solver classes and combining it with one or more mixins.
#    - Several solver implementations are already readily available in PINA and can be used directly or extended to define more specialized training strategies.

# ## What's Next?
# 
# Congratulations on completing the tutorial on solver classes! Now that you have a solid foundation, here are a few directions you can explore:
# 
# 
# 1. **Physics-Informed Solvers**: Try to implement your own physics-based solver. Can you do it?
# 
# 2. **Multi-Model Solvers**: Take it to the next level by exploring multi-model solvers. You could implement and train models that combine the strengths of multiple neural networks.
# 
# 3. **...and many more!**: There are countless directions to further explore, try to look at our `solver` for example!
# 
# For more resources and tutorials, check out the [PINA Documentation](https://PINA-org.github.io/PINA/).
