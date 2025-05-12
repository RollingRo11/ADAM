# ADAM
Python implementation of "ADAM: A Method for Stochastic Optimization" (P. Kingma & Ba).

This is just a small feed-forward model that aims to place URLs into 3 groups: **benign**, **phishing**, or **defacement**.

Dataset from [here!](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset/data)

Read the paper [here!](https://arxiv.org/abs/1412.6980)

## Paper Implementation Overview:

The important part of the implementation lies in [`optimizer.py`](). I used Claude to generate method stubs and comment documentation, so I could approach this like an assignment! The optimizer is meant to be a drop-in replacement, so you can swap out my optimizer for the official PyTorch one, or any other one in [`main.py`]()!

**The `update_parameter()` function strictly follows the paper, here's how I implemented each step:**
1) $m_t \leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \text{ (Update biased first moment estimate)}$
```Python
self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
```

2) $v_t \leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2 \text{ (Update biased second raw moment estimate)}$
```Python
self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad**2)
```

3) $\hat{m}_t \leftarrow m_t / (1-\beta_1^t) \text{ (Compute bias-corrected first moment estimate)}$
```Python
        m_hat = self.m[param_id] / (1 - (self.beta1**self.t))
```

4) $\hat{v}_t \leftarrow v_t / (1-\beta_2^t) \text{ (Compute bias-corrected second raw moment estimate)}$
```Python
v_hat = self.v[param_id] / (1 - (self.beta2**self.t))
```

5) $\theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) \text{ (Update parameters)}$
```Python
param.data -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.eps)
```


## Statistics:

### Network with `torch.optim.SGD` (learning_rate=0.01)
```
Epoch 1: loss: 0.6662 | accuracy: 73.93%
Epoch 2: loss: 0.4643 | accuracy: 82.17%
Epoch 3: loss: 0.4094 | accuracy: 84.13%
Epoch 4: loss: 0.3716 | accuracy: 85.79%
Epoch 5: loss: 0.3464 | accuracy: 86.88%
Test accuracy: 88.97%
```

### Network with `torch.optim.Adam` (learning_rate=0.001)
```
Epoch 1: loss: 0.2212 | accuracy: 92.14%
Epoch 2: loss: 0.1381 | accuracy: 95.59%
Epoch 3: loss: 0.1210 | accuracy: 96.19%
Epoch 4: loss: 0.1112 | accuracy: 96.53%
Epoch 5: loss: 0.1049 | accuracy: 96.77%
Test accuracy: 96.97%
```

### Network with my implementation of Adam
```
Epoch 1: loss: 0.2176 | accuracy: 92.31%
Epoch 2: loss: 0.1392 | accuracy: 95.51%
Epoch 3: loss: 0.1212 | accuracy: 96.21%
Epoch 4: loss: 0.1106 | accuracy: 96.56%
Epoch 5: loss: 0.1042 | accuracy: 96.79%
Test accuracy: 97.04%
```
