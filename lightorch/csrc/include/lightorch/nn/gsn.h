// Optimized implementation of Gram Schmidt Networks 
#include <torch/extensions.h>
#include <math.h>
at::Tensor gsn() {

}

at::Tensor gram_schmidt (Layers) {

}


// Defining the Frontibier inner product for linear transformation space
float inner (const at::Tensor& L_1, const at::Tensor& L_2) {
    return torch::trace(torch::addmm(L_1.transpose({-1 , -2}).conj(), L_2));
}

// Defining the generalized norm for any linear space
float norm (const at::Tensor& L_1, const at::Tensor& L_2) {
    return sqrt(inner(L_1, L_2));
}

// Defining the projection operation proj_u v
at::Tensor proj(const at::Tensor& u, const at::Tensor& v) {
    return torch::mul((inner(v, u)/inner(u, u)), u);
}

