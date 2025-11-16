#include "adam.hpp"
#include "../config.hpp"
#include <stdexcept>
#include <cmath>
#include "adam.cuh"

Adam::Adam(float beta1, float beta2, float eps)
    : beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      m_(nullptr),
      v_(nullptr),
      t_(0),
      last_num_elements_(0) {
}

Adam::~Adam() {
    if (m_) {
        zerollm_backend::free(m_);
    }
    if (v_) {
        zerollm_backend::free(v_);
    }
}

void Adam::step(float* param, const float* grad, int num_elements, float learning_rate) {
    if (num_elements <= 0) {
        throw std::invalid_argument("Number of elements must be positive");
    }
    
    // 如果参数数量发生变化，重新分配缓冲区
    if (last_num_elements_ != num_elements) {
        if (m_) {
            zerollm_backend::free(m_);
            m_ = nullptr;
        }
        if (v_) {
            zerollm_backend::free(v_);
            v_ = nullptr;
        }
        
        m_ = (float*)zerollm_backend::malloc(num_elements * sizeof(float));
        v_ = (float*)zerollm_backend::malloc(num_elements * sizeof(float));
        
        // 初始化缓冲区为0
        zerollm_backend::memset(m_, 0, num_elements * sizeof(float));
        zerollm_backend::memset(v_, 0, num_elements * sizeof(float));
        
        last_num_elements_ = num_elements;
    }
    
    // 增加时间步
    t_++;
    
    // 计算偏差修正项
    float bias_correction1 = 1.0f - powf(beta1_, static_cast<float>(t_));
    float bias_correction2 = 1.0f - powf(beta2_, static_cast<float>(t_));
    
    // 更新一阶矩和二阶矩估计
    adam_step<float>(param, grad, m_, v_, num_elements, learning_rate, beta1_, beta2_, eps_, bias_correction1, bias_correction2, 0);
    
    zerollm_backend::device_synchronize();
}

void Adam::zero_grad() {
    if (m_ && v_) {
        zerollm_backend::memset(m_, 0, last_num_elements_ * sizeof(float));
        zerollm_backend::memset(v_, 0, last_num_elements_ * sizeof(float));
    }
    t_ = 0;
}