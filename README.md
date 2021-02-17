# Gaussian Covariate Model and applications
Repository for the paper [*Capturing the learning curves of generic features maps for realistic data sets with a teacher-student model*](https://arxiv.org/abs/2102.08127).

# Structure

 We provide a couple of guided examples to help the reader reproduce the figures of the paper. The key ingredients are:

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```/state_evolution/``` | Out-of-the-box package for solving saddle-points equations for classification and regression tasks,      |                               |
| ```how_to.ipynb``` | Notebook with a step-by-step explanation on how to use the `state_evolution` package.                                     |
| ```/real_data/mnist_scattering.ipynb``` | Notebook reproducing real-data curves, see Fig. 4 of the paper.  |
| ```/gan_data/synthetic_data_pipeline.ipynb ```         | Notebook explaining pipeline to assign labels for GAN generated data.                                                               |
| ```/gan_data/monte_carlo.ipynb ```         | Notebook explaining how to estimate population covariances for features from GAN generated data.                                                               |
| ```/gan_data/learning_curves.ipynb ```         | Notebook reproducing learning curves for GAN generated data, see Fig. 3 of the paper.                                                              |

The notebooks are self-explanatory. You will also find some auxiliary files such as `simulations.py` in `/real_data` wrapping the code for running the simulations, and `dcgan.py`, `teachers.py`, `teacherutils.py`, `simulate_gan.py` in `/gan_data/` wrapping the different torch models for the pre-trained generators and teachers.

Note that for running the examples in ```/gan_data``` you will need the weights of the generator, the teacher and the covariances. A folder can be downloaded [here](https://drive.google.com/file/d/1XMm5NDFm3Ol0eqLjvgN5XriQcSNtI3ZN/view?usp=sharing) in a single folder ```/data```.

# Reference

[1]: *Capturing the learning curves of generic features maps for realistic data sets with a teacher-student model*,
B Loureiro, C Gerbelot, H Cui, S Goldt, F Krzakala, M Mézard, L Zdeborová, [arXiv: 2102.08127](https://arxiv.org/abs/2102.08127) [stat.ML]
