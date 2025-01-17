## Experiment on mnist images

Train the framework for generation using `train.py`

or use pre-trained framework located in `./models`

Then generate natural adversaries using `test.py`

Classifiers: 
- Random Forest (90.45%), `--classifier rf`
- LeNet (98.71%), `--classifier lenet`

Algorithms: 
- iterative stochastic search, `--iterative` 
- hybrid shrinking search (default)

Output samples are located in `./examples`

Defensive distillation:
We have referred the following paper:
Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks by Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami

Link to distillation code: https://github.com/carlini/nn_robust_attacks
