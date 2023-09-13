# CaloClouds II Model for Fast Calorimeter Simulation

PyTorch implementation of the CaloClouds II Model introduced in *CaloClouds II: Ultra-Fast Geometry-Independent Highly-Granular Calorimeter Simulation* ([arXiv:2309.05704](https://arxiv.org/abs/2309.05704)).

The model is an evolution of the CaloClouds model published in ([arXiv:2305.04847](https://arxiv.org/abs/2305.04847)).

---

The CaloClouds II Model generates photon showers in point cloud format with up to 6000 points per shower for an energy range between 10 and 90 GeV. 
It consists out of multiple sub-generative models, including the PointWise Net trained as a score-based generative model and a normalizing flow for generating shower observables. 
Further, consistency distllation is implemented to distill a trained CaloClouds II diffusion model into a consistency model dubbed CaloClouds II (CM).
The training data is generated with a Geant4 simulation of the planned electromagnetic calorimeter of the International Large Detector (ILD).

---

The CaloClouds II Pointwise Net Diffusion Model can be trained via `python training.py` using the default parameters set in [`config.py`](./configs.py).

A trained CaloClouds II Diffusion model can be distilled into a consistency model via `python distillation.py`.

The Shower Flow is trained via the notebook [`ShowerFlow.ipynb`](./ShowerFlow.ipynb).

The polynomial fits for the occupancy calculations are performed in [`occupancy_scale.ipynb`](./occupancy_scale.ipynb).

An outline of the sampling process for both CaloClouds II and CaloClouds II (CM) can be found in [`generate.py`](./evaluation/generate.py).

The timing of the models is benchmarked with [`calc_timing.py`](./calc_timing.py)

---

The training dataset is available under the link: https://syncandshare.desy.de/index.php/s/XfDwx33ryERwPdi

---

Code references:
- The code for training the score-based model is based on: https://github.com/crowsonkb/k-diffusion
- The consistency distillation is based on: https://github.com/openai/consistency_models/
- The PointWise Net is adapted from: https://github.com/luost26/diffusion-point-cloud
- Code base for our CaloClouds (1) model: https://github.com/FLC-QU-hep/CaloClouds
