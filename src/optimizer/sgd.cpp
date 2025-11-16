#include "sgd.hpp"
#include "../config.hpp"
#include <stdexcept>
#include <cstring>

#include "sgd.cuh"

SGD::SGD(float momentum)
    : momentum_(momentum),
      velocity_(nullptr),
      use_momentum_(momentum > 0.0f),
      last_num_elements_(0) {
}

SGD::~SGD() {
    if (velocity_) {
        zerollm_backend::free(velocity_);
    }
}

void SGD::step(float* param, const float* grad, int num_elements, float learning_rate) {
    if (num_elements <= 0) {
        throw std::invalid_argument("Number of elements must be positive");
    }
    
    // 如果参数数量发生变化，重新分配缓冲区
    if (last_num_elements_ != num_elements) {
        if (velocity_) {
            zerollm_backend::free(velocity_);
            velocity_ = nullptr;
        }
        
        if (use_momentum_) {
            velocity_ = (float*)zerollm_backend::malloc(num_elements * sizeof(float));
            zerollm_backend::memset(velocity_, 0, num_elements * sizeof(float));
        }
        
        last_num_elements_ = num_elements;
    }
    
    sgd_step<float>(param, grad, velocity_, num_elements, learning_rate, momentum_, 0.0f, 0);
    
    zerollm_backend::device_synchronize();
}

void SGD::zero_grad() {
    if (use_momentum_ && velocity_) {
        zerollm_backend::memset(velocity_, 0, last_num_elements_ * sizeof(float));
    }
}