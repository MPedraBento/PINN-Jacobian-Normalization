import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils import parameters_to_vector
import numpy as np
import types

class NegativeSigmoid(nn.Module):
    def forward(self, x):
        return -torch.sigmoid(x)

class PINN(nn.Module):

    def __init__(self, units, num_layers, num_outputs, lr, pinn_loss, output_sigmoid=False, C=None):
        super().__init__()
        
        self.units = units
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.output_sigmoid = output_sigmoid
        self.model = self.build_model()
        self.lr = lr
        self.pinn_loss = types.MethodType(pinn_loss, self)
        self.C = C
        if C is not None:
            self.optimizer = optimizer = torch.optim.AdamW(list(self.parameters()) + [C], lr=lr)
        else:
            self.optimizer = optimizer = torch.optim.AdamW(list(self.parameters()), lr=lr)

    def mask_fn(self, lbd):
        """
        Computes the self-adaptive mask function, considered here to be a sigmoid function.

        Parameters:
            lbd (float): A weight, or an array of them.
        
        Returns:
            float: The weights lbd transformed by a sigmoid function.
        """
        return torch.sigmoid(lbd)

    def deriv_mask_fn(self, lbd):
        """
        Computes the derivative of the self-adaptive mask function, considered here to be a sigmoid function.

        Parameters:
            lbd (float): A weight, or an array of them.
        
        Returns:
            float: The weights lbd transformed by the derivative of a sigmoid function.
        """
        return torch.sigmoid(lbd) * (1.0 - torch.sigmoid(lbd))

    def build_model(self):

        layers = []
    
        layers.append(torch.nn.Linear(1, self.units))
        layers.append(torch.nn.GELU())
        
        for _ in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(self.units, self.units))
            layers.append(torch.nn.GELU())
        
        layers.append(torch.nn.Linear(self.units, self.num_outputs))
        
        if self.output_sigmoid:
            layers.append(NegativeSigmoid())
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def train_step(self, x, lbd_r=None, lbd_0=None):
    
        self.train()

        if self.C is None and lbd_r is None and lbd_0 is None:
            ode_loss, x0_loss, total_loss = self.pinn_loss(x)
        elif lbd_r is not None and lbd_0 is None:
            ode_loss, x0_loss, total_loss, residual, lbd_r = self.pinn_loss(x, lbd_r)
        elif lbd_r is not None and lbd_0 is not None:
            ode_loss, x0_loss, total_loss, ode_unmasked, x0_unmasked, lbd_r, lbd_0 = self.pinn_loss(x, lbd_r, lbd_0)
        else:
            ode_loss, x0_loss, xf_loss, total_loss = self.pinn_loss(x)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if lbd_r is not None and lbd_0 is None:
            gamma = 0.999
            eta_ast = 0.01
            lbd_r = gamma * lbd_r + eta_ast * torch.abs(torch.squeeze(residual)) / torch.max(torch.abs(residual))
            return {
                "total_loss": total_loss.item(),
                "ode_loss": ode_loss.item(),
                "x0_loss": x0_loss.item(),
                "lbd_r": lbd_r
            }

        elif lbd_r is not None and lbd_0 is not None:
            rho_r = 0.005
            rho_0 = 0.005
            lbd_r += rho_r * self.deriv_mask_fn(lbd_r) * torch.squeeze(ode_unmasked)
            lbd_0 += rho_0 * self.deriv_mask_fn(lbd_0) * x0_unmasked
            return {
                "total_loss": total_loss.item(),
                "ode_loss": ode_loss.item(),
                "x0_loss": x0_loss.item(),
                "lbd_r": lbd_r,
                "lbd_0": lbd_0
            }
        
        elif self.C is not None:
            return {
                "total_loss": total_loss.item(),
                "ode_loss": ode_loss.item(),
                "x0_loss": x0_loss.item(),
                "xf_loss": xf_loss.item(),
                "C": self.C.item()
            }
            
        else:
            return {
                    "total_loss": total_loss.item(),
                    "ode_loss": ode_loss.item(),
                    "x0_loss": x0_loss.item()
                }
            

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    
    def compute_dy_dw_eigenvalue(self, inputs):
        # Get all parameters as a single vector
        parameters = [p for p in self.model.parameters() if p.requires_grad]

        y = self.model(inputs)
        dy_dw = autograd.grad(y / y.shape[0], parameters, grad_outputs=torch.ones_like(y), retain_graph=True)
        dy_dw = torch.cat([g.view(-1) for g in dy_dw])

        return torch.sum(dy_dw**2).item()


    def compute_largest_hessian_eigenvalue(self, x, num_iters=100):
        """
        Computes the largest eigenvalue of the Hessian of the loss w.r.t. model parameters
        using the power iteration / Lanczos method (matrix-free).
        
        Args:
            x: Input tensor to your model
            num_iters: Number of power iterations to run (default: 20)
    
        Returns:
            Largest eigenvalue (scalar)
        """
        self.model.eval()
        
        # Flattened list of model parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
    
        # Initialize a random unit vector (on the same device as model)
        v = torch.randn(n_params, device=x.device)
        v = v / v.norm()
    
        for _ in range(num_iters):
            # Compute loss and first gradient
            loss, *_ = self.pinn_loss(x)
            grads = autograd.grad(loss, params, create_graph=True)
            grads_flat = torch.cat([g.view(-1) for g in grads])
    
            # Compute Hessian-vector product: ∇²L · v
            Hv = autograd.grad(grads_flat, params, grad_outputs=v, retain_graph=True)
            Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv])
    
            # Normalize for next iteration
            v = Hv_flat / Hv_flat.norm()

            #output_mean = torch.mean()
            output = self.model(x)
       
        # Final Rayleigh quotient approximation: vᵀHv
        eigenvalue = v @ Hv_flat
        return eigenvalue.item()