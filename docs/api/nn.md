# complex
Defines two modules for the real and complex space.
```python 
from lightorch.nn.complex import Complex
from torch import nn
layer = Complex(nn.Conv2d(...)) #input(complex) -> output(complex)
```
# criterions
A module of standardized criterions to use with the framework.

- $I$: Input
- $O$: Output
- $\psi_p$: Features extracted from the layer $p$ from a `FeatureExtractor`


## LighTorchLoss
Base module to create criterions.

```python 
from lightorch.nn.criterions import LighTorchLoss

class Loss(LighTorchLoss):
    def __init__(self, alpha: Sequence[float], *args) -> None:
        labels = ['Criterion1', 'Criterion2', 'Criterion3']
        super().__init__(
            labels = labels,
            factors = {name: factor for name, factor in zip(labels, alpha)}
        )
        self.alpha = alpha

        ... # Continue with dependencies

    def forward(self, **kwargs) -> Tuple[Tensor, ...]:
        # return each loss term without factor terms
        return L1, L2, L3, self.alpha[0]*L1 + self.alpha[1]*L2  + self.alpha[2]*L3
```

## CrossEntropyLoss
Cross Entropy Loss module, the same as `nn.CrossEntropyLoss` adjusted for LighTorch.

```python 
>> from lightorch.nn.criterions import CrossEntropyLoss
>> input, target = torch.randn(32, 1), torch.randn(32, 1)
>> loss = CrossEntropyLoss(factor = 3)
>> loss(input = input, target = target)
```


## MSELoss
Mean Squared Error loss module, the same as `nn.MSELoss` adjusted for LighTorch.

```python 
>> from lightorch.nn.criterions import MSELoss
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> loss = MSELoss(factor = 2)
>> loss(input = input, target = target)
```

## ELBO


- `beta (float)`: Multiplication factor of the Kullback-Leibler Divergence (KL Divergence) for the $\beta$-VAE training, $\beta$=1 is default VAE training.
- `reconstruction_loss (LighTorchLoss)`: LighTorchLoss module to represent the reconstruction term in the loss function.

```python 
>> from lightorch.nn.criterions import ELBO, MSELoss
>> mu, logvar = torch.randn(32, 32), torch.randn(32, 32)
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> beta = 1e-4 # For \beta-VAE
>> loss = ELBO(beta = beta, reconstruction_loss = MSELoss())
>> loss(input = input, target = target, mu = mu, logvar = logvar)

```

## StyleLoss
$

\begin{equation}
    \mathcal{L}_{style} := \sum_{p \in P} \frac{||(\psi_p^{I(\theta)})^T(\psi_p^{I(\theta)}) - (\psi_p^{O})^T(\psi_p^{O})||_1}{F_p} 
\end{equation}

$

- `feature_extractor (FeatureExtractor)`: Feature extractor module that return features from hidden layers.  
- `factor (float)`: Multiplication factor of the Gram-Matrix based style loss.

```python 
>> from lightorch.nn.criterions import StyleLoss
>> from lightorch.nn.features import FeatureExtractor 
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> beta = 1e-4 # For \beta-VAE
>> loss = 
>> loss(input = input, target = target, feature_extractor = True) # feature_extractor = True assumes input: $I$, else the input is already input: $\phi$

```
## PerceptualLoss
$

\begin{equation}
    \mathcal{L}_{perceptual} := \sum_{p \in P} \frac{||\psi_p^{I(\theta)} - \psi_p^{O}||_1}{N_{\psi_{p}}}
\end{equation}

$

- $N_{\psi_{p}}$: C * H * W of the p-th feature space.
- $\psi_{p}$: P-th feature space output.

`Arguments`
- `feature_extractor (FeatureExtractor)`: Feature extractor module that return features from hidden layers.  
- `factor (float)`: Multiplication factor of the Gram-Matrix based style loss.

```python 
>> from lightorch.nn.criterions import StyleLoss
>> from lightorch.nn.features import FeatureExtractor 
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> beta = 1e-4 # For \beta-VAE
>> loss = 
>> loss(input = input, target = target, feature_extractor = True) # feature_extractor = True assumes input: $I$, else the input is already input: $\phi$

```
## PeakSignalNoiseRatio
$
\begin{equation}
    \mathcal{L}_{PSNR} := 10 \log_{10}\left(\frac{MAX^2}{MSE}\right)
\end{equation}

$
`Arguments`
- `factor (float)`: Multiplication factor of the loss.
- `max (float)`: Maximum value of the input and target space.

```python 
>> from lightorch.nn.criterions import PeakSignalNoiseRatio
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> loss = PeakSignalNoiseRatio(max = 1, factor = 1) 
>> loss(input = input, target = target)
```

## TV
$
\begin{equation}
    \mathcal{L}_{tv} = \sum_{i,j} \left(|| I^{i, j+1} - I^{i, j}||_1 + || I^{i+1, j} - I^{i, j}||_1 \right)
\end{equation}
$
`Arguments`
- `factor (float)`: Multiplication factor of the loss.

```python 
>> from lightorch.nn.criterions import TV
>> input = torch.randn(32, 10, 10)
>> loss = TV(factor = 1)
>> loss(input = input)
```


## Loss
Module to join multiple criterions at once.

`Arguments`
- `*loss`: LighTorchLoss initialized modules that will be concatenated into the Loss.

```python 
>> from lightorch.nn.criterions import MSELoss, TV, Loss
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> loss = Loss(MSELoss(factor = 1), TV(factor = 1))
>> loss(input = input, target = target)
```

## LagrangianFunctional
Object for interpretative constraint optimization with the Lagrangian Functional.

`Arguments`
- `f (Callable)`: Main scalar field to optimize.
- `*g`: Constraints to the function.
- `lambd (Tensor)`: Sequence of scalar multipliers. 
- `f_name (str)`: Name of the main scalar field. 
- `g_name (Sequence[str])`: Sequence of the names of the constraints. 


```python 
>> from lightorch.nn.criterions import MSELoss, TV, LagrangianFunctional
>> input, target = torch.randn(32, 10, 10), torch.randn(32, 10, 10)
>> loss = LagrangianFunctional(f = MSELoss(factor = 1), TV(factor = 1))
>> loss(input = input, target = target)
```


# dnn
Deep Neural Netowrk instances:

## DeepNeuralNetwork
```python 
from lightorch.nn.dnn import DeepNeuralNetwork
from torch.nn import ReLU, Sigmoid
from torch import nn, Tensor
import torch 

sample_input: Tensor = torch.randn(32, 10) # batch size, input_size

dnn = DeepNeuralNetwork(
    in_features = 10,
    layers = (20, 30, 20, 1),
    activations = (ReLU(), None, ReLU(), Sigmoid())
)
# Creates Neural network with input size 10
# input, hidden and output layer with number of 
# perceptrons: 20 (input), 30 (hidden), 20 (hidden), 1 (output)
# with activations functions after each layer
#

dnn(sample_input) #-> output (32, 1)

```

# fourier
This module has been made to create highly paralelizable convolutions with the convolution theorems:
1. Convolution Theorem:
$
\begin{equation}
    \mathcal{F}(f * g) = \mathcal{F}(f) \odot \mathcal{F}(g)
\end{equation}
$

2. Deconvolution implication:
$
\begin{equation}
    \frac{\mathcal{F}(f*g)}{\mathcal{F}(g) + \epsilon} = \frac{\mathcal{F}(f) \odot \mathcal{F}(g)}{\mathcal{F}(g) + \epsilon} \approx \mathcal{F}(f)
\end{equation}
$

## FourierConv
This module expands an input signal channels dimension from in_channels to out_channels with a non-learnable convolution with kernel size: 1 and stride: 1. This computation is efficiently accomplished taking advantage of its highly paralelizable nature. After that the input signal is segmented into sub-signals of size kernel_size, then the convolution in the fourier space is computed with a trainable weight and bias channel-wise.
$\text{Expand convolution} \to \text{Patch} \to \text{Fourier Space convolution}: \mathcal{F}(I) \odot \mathcal{F}(W) + \mathcal{F}(b)$

```python 
from lightorch.nn.fourier import FourierConv2d
from torch import nn, Tensor

sample_input: Tensor = torch.randn(32, 3, 256, 256) # batch size, input_size

model = nn.Sequential(
    FourierConv2d(3, 10, 5, 1, pre_fft = True),
    FourierConv2d(10, 20, 5, 1, post_ifft = True)
)

model(sample_input) #-> output (32, 20, 256, 256)

```

## FourierDeconv
$\text{Expand convolution} \to \text{Patch} \to \text{Fourier Space deconvolution}: \frac{\mathcal{F}(I)}{\mathcal{F}(W)}$

```python 
from lightorch.nn.fourier import FourierConv2d
from torch import nn, Tensor

sample_input: Tensor = torch.randn(32, 3, 256, 256) # batch size, input_size

model = nn.Sequential(
    FourierDeconv2d(3, 10, 5, 1, pre_fft = True),
    FourierDeconv2d(10, 20, 5, 1, post_ifft = True)
)

model(sample_input) #-> output (32, 20, 256, 256)

```

# functional

# kan

# monte_carlo
Randomly drops features from the input tensor and passes it through the fc layer n times, takes the mean of the output.
```python 
from lightorch.nn.monte_carlo import MonteCarloFC
from lightorch.nn.dnn import DeepNeuralNetwork
from torch import nn, Tensor

sample_input: Tensor = torch.randn(32, 10) # batch size, input_size

model = MonteCarloFC(
    fc_layer = DeepNeuralNetwork(
        in_features = 10,
        (20, 20, 1),
        (nn.ReLU(), nn.ReLU(), nn.Sigmoid())
    ),
    dropout = 0.5,
    n_sampling = 50
)

model(sample_input) #-> output (32, 1)

```
# normalization
Module non-built in normalization methods that retrieved good results in other researchs.

```python 
from lightorch.nn.normalization import RootMeanSquaredNormalization
from torch import nn, Tensor

sample_input: Tensor = torch.randn(32, 20, 10) # batch size, sequence_length, input_size

norm = RootMeanSquaredNormalization(dim = 10)

model(sample_input) #-> output (32, 20, 10)

```

# partial
Partial convolutions from [this research](https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf) redefined.


$
O = W^T (X \odot M) \frac{sum(1)}{sum(M)} + b
$


$
m' = \begin{cases} 
1 & \text{if } \sum(M) > 0 \\
0 & \text{otherwise}
\end{cases}
$

```python 
from lightorch.nn.partial import PartialConv2d
from torch import nn, Tensor

sample_input: Tensor = torch.randn(32, 3, 256, 256) # batch size, channels, height, width
mask_in: Tensor = sample_input()

model = nn.Sequential(
    PartialConv2d(in_channels = 3, out_channels = 5, 3, 1, 1),
    PartialConv2d(in_channels = 5, out_channels = 5, 3, 1, 1)
)

model(sample_input, mask_in) #-> output (32, 5, 256, 256)


```


