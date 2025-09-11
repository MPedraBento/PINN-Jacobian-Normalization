import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
import numpy as np
import types

class LearnedGelu(nn.Module):
    """
    A learnable version of the GELU activation function.

    This module introduces a learnable parameter 'slope' which scales the input
    before applying the GELU function. This allows the network to adjust the
    shape of the activation function during training.
    """
    def __init__(self, slope=1.0):
        """
        Initializes the LearnedGelu module.

        Args:
            slope (float, optional): The initial value for the learnable slope.
                                     Defaults to 1.0, which makes it behave
                                     like a standard GELU initially.
        """
        super(LearnedGelu, self).__init__()
        # nn.Parameter wraps a tensor to make it a learnable parameter of the module.
        # The optimizer will update this value during backpropagation.
        self.slope = nn.Parameter(torch.Tensor(1).fill_(slope))

    def forward(self, input_tensor):
        """
        Applies the learned GELU activation.

        Args:
            input_tensor (torch.Tensor): The input tensor from the previous layer.

        Returns:
            torch.Tensor: The output tensor after applying the activation.
        """
        # Multiply the input by the learnable slope, then apply the standard GELU function.
        return F.gelu(self.slope * input_tensor)

class PINN(nn.Module):

    def __init__(self, units, num_layers, pinn_loss):
        super().__init__()
        
        self.units = units
        self.num_layers = num_layers
        self.model = self.build_model()
        self.pinn_loss = types.MethodType(pinn_loss, self)
        # After model
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)

    def build_model(self):

        layers = []
    
        layers.append(torch.nn.Linear(2, self.units))
        layers.append(LearnedGelu())
        
        for _ in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(self.units, self.units))
            layers.append(LearnedGelu())
        
        layers.append(torch.nn.Linear(self.units, 1))
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def train_step(self, x):


        self.train()
        
        ode_loss, x0_loss, total_loss = self.pinn_loss(x)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5.0)
        self.optimizer.step()
        #self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "ode_loss": ode_loss.item(),
            "x0_loss": x0_loss.item()
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


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
            *_, loss = self.pinn_loss(x)
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