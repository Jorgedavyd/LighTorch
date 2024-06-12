// Optimized implementation of the fourier layer
#include <torch/extension.h>
#include <complex>
#include <iostream>
#include <vector>

namespace F = torch::nn::functional;

// Defining the derivatives for the backward pass


// Defining the Backward pass
std::vector<torch::Tensor> fourier_conv_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const bool& pre_fft,
    const bool& post_ifft
) {
    torch::Tensor grad_input, grad_weight, grad_bias, grad_output_fft;

    struct out = {};
    if (pre_fft) {
        // Compute the derivative of the fourier transform
    }
    // Compute the default derivative

    if (post_ifft) {
        // Compute the derivative of the inverse fourier transform
    }
}

// Defining the Forward pass
torch::Tensor fourier_conv_forward (
    at::Tensor& input,
    at::Tensor& weight,
    at::Tensor& bias,
    const bool& pre_fft,
    const bool& post_ifft
) {
    at::Tensor out;
    // Assertions 
    TORCH_CHECK(input.dim() >= 3 && input.dim() <= 5, 'Input must be over 3 dimensions (min 1d_conv: (batch, channel, size)), and under 5 dimensions (max 3d_conv: (batch, channel, frame, height, width))');
    TORCH_CHECK(input.dim() - 2 == weight.dim(), 'Input dimension must be equal to the weight dimension');
    TORCH_CHECK(bias.dim() == 1, 'Bias should be of size 1, don\'t reshape.');
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(weight.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(bias.device().type() == at::DeviceType::CPU);
    
    // define the dimensionality 
    if (pre_fft || post_ifft) {
        // Defining the dimensions for the fft and ifft
        std::vector<int64_t> dim;
        // Compute the fft of the input
        switch (input.dim() - 2) {
            case 1: const struct dim = {-1}; break;
            case 2: const struct dim = {-1, -2}; break;
            case 3: const struct dim = {-1, -2, -3}; break;
            default: TORCH_CHECK(false, 'Unsupported in put dimension'); break;
            }
        
        if (pre_ifft) {
            // To Fourier space
            input = torch::fft::fftn(input, {}, dim);
            weight = torch::fft::fftn(weight, {}, dim);
            bias = torch::fft::fftn(bias, {}, dim = {-1});
        } 
    }

    // Convolution in Fourier Space
    out = input * weight;

    // Add the bias term
    if (bias.defined()) {
        out += bias.view(1, -1, 1, 1);
    }

    if (post_ifft) {
        // Compute the ifft
        out = torch::fft::ifftn(out, {}, dim);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def('fourier_conv_forward', &fourier_conv_forward, 'Fourier Convolutions Forward'),
    m.def('fourier_conv_backward', &fourier_conv_backward, 'Fourier Convolutions Backward')
}

