#include "adamw.hpp"
#include "../config.hpp"
#include <stdexcept>
#include <cmath>
#include "adamw.cuh"

AdamW::AdamW(float weight_decay, float beta1, float beta2, float eps)
    : Adam(beta1, beta2, eps),
      weight_decay_(weight_decay) {
}

void AdamW::step(float* param, const float* grad, int num_elements, float learning_rate) {
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
    
    // AdamW更新规则:
    // 1. 应用权重衰减到参数: param = param - learning_rate * weight_decay * param
    // 2. 更新一阶矩和二阶矩估计
    // 3. 使用修正后的矩估计更新参数
    
    adamw_step<float>(param, grad, m_, v_, num_elements, learning_rate, weight_decay_, beta1_, beta2_, eps_, bias_correction1, bias_correction2, 0);
    
    zerollm_backend::device_synchronize();
}