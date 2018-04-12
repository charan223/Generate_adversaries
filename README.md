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
