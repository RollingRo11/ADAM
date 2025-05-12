import torch
import torch.nn as nn


class Adam(nn.Module):
    def __init__(self, params, lr=0.01):
        super().__init__()
        self.params = list(params)
        self.learning_rate = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.m = {}
        self.v = {}
        self.initialize_state()

    def initialize_state(self):
        """
        Initialize momentum (m) and velocity (v) states for each parameter.

        This method should create zero-filled arrays matching the shape of each parameter
        for both first moment (momentum) and second moment (velocity) estimates.

        Example implementation would initialize:
        - self.m[param_id] = np.zeros_like(param.data)
        - self.v[param_id] = np.zeros_like(param.data)
        for each parameter in self.params.
        """
        for i, param in enumerate(self.params):
            self.m[i] = torch.zeros_like(param.data)
            self.v[i] = torch.zeros_like(param.data)

    def step(self):
        """
        Performs a single optimization step using the Adam update rule.

        This method should:
        1. Increment the time step counter (self.t)
        2. For each parameter:
           - Update biased first moment estimate (momentum)
           - Update biased second moment estimate (velocity)
           - Compute bias-corrected first moment estimate
           - Compute bias-corrected second moment estimate
           - Update parameters using the Adam formula:
             θ = θ - learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)

        Returns:
            None
        """

        self.t += 1
        for i, param in enumerate(self.params):
            if param is None:
                continue
            self.update_parameter(param, i)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.

        This method should set the gradient (grad attribute) of each parameter to zero.
        In NumPy implementation, this would involve setting the gradient arrays to zeros
        for each parameter that is being optimized.

        Returns:
            None
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def update_parameter(self, param, param_id):
        """
        Update a single parameter using the Adam update rule.

        The Adam update follows these equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)

        where:
        - g_t is the gradient at time t
        - m_t is the biased first moment estimate
        - v_t is the biased second moment estimate
        - m_hat is the bias-corrected first moment estimate
        - v_hat is the bias-corrected second moment estimate
        - theta_t is the parameter value at time t

        Args:
            param: The parameter to update (should have data and grad attributes)
            param_id: Unique identifier for the parameter in the state dictionaries

        Returns:
            None
        """
        grad = param.grad.data
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad**2)

        m_hat = self.m[param_id] / (1 - (self.beta1**self.t))
        v_hat = self.v[param_id] / (1 - (self.beta2**self.t))

        param.data -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.eps)
