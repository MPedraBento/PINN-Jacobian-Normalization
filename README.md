# Introduction

We propose a simple, hyperparameter-free method to address stiffness in Physics-Informed Neural Networks (PINNs) by normalizing loss residuals with the Jacobian. We show that Jacobian-based normalization improves gradient descent and validate it on benchmark stiff ordinary differential equations. We then apply it to a realistic system: the stiff Boltzmann equations (BEs) governing weakly interacting massive particle (WIMP) dark matter (DM). Our approach achieves higher accuracy than attention mechanisms previously proposed for handling stiffness, recovering the full solution where prior methods fail. This is further demonstrated in an inverse problem with a single experimental data point - the observed DM relic density - where our inverse PINNs correctly infer the cross section that solves the BEs in both Standard and alternative cosmologies.

<p align="center">
  <img src="https://github.com/MPedraBento/PINN-Jacobian-Normalization/blob/main/plots/FwdComparison-SigmoidTrue.png" width="300" />  &nbsp  &nbsp &nbsp
  <img src="https://github.com/MPedraBento/PINN-Jacobian-Normalization/blob/main/plots/InvPreds.png" width="300" /> 
</p>

# Notebooks
utils.py
pinn.py

## Jacobian Normalization and Its Effect on PINN Convergence
Toy1-SimplestODE.ipynb <br>
Toy2-NonHomogeneous.ipynb <br>
Toy3-NonLinear.ipynb <br>

## A Concrete Application: WIMP Dark Matter in Alternative Cosmologies
forward-Jacobian.ipynb <br>
forward-RBA.ipynb <br>
forward-Soft.ipynb <br>
inverse-C.ipynb <br> <br>

All the aforementioned notebooks generate the necessary files to draw the plots in ResultsSection2.ipynb and ResultsSection3.ipynb.
