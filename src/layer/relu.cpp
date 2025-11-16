#include "relu.hpp"
#include <stdexcept>
#include "config.hpp"
#include "relu.cuh"

/**
 * @brief ReLU构造函数
 */
ReLU::ReLU() : Activation() {}

/**
 * @brief ReLU析构函数
 */
ReLU::~ReLU() {}

/**
 * @brief 前向传播 RELU 
 * 
 * 执行ReLU激活函数: f(x) = max(0, x)
 * @param output 输出数据存放指针 [batch_size, features]
 * @param input 输入数据存放指针 [batch_size, features]
 * @param batch_size 批次大小
 * @param features 特征数量
 */
void ReLU::forward(float* output, const float* input, int batch_size, int features) {
    this->input_ = input;
    this->output_ = output;
    this->last_batch_size_ = batch_size;
    this->features_ = features;
    // input_shape = (batch_size, features)
    int total_elements = batch_size * features;

    relu_forward(input, output, total_elements, 0);

    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("前向传播错误");
}

/**
 * @brief 反向传播
 * 
 * 计算ReLU激活函数的梯度
 * @param d_input 输入梯度 [batch_size, features]
 * @param d_output 输出梯度 [batch_size, features]
 */
void ReLU::backward(float* d_input, const float* d_output) {
    if (input_ == nullptr) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    int total_elements = last_batch_size_ * features_;
    relu_backward(input_, d_output, d_input, total_elements, 0);
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("反向传播错误");
}

/**
 * @brief 清零梯度
 * 
 * ReLU层本身没有可学习参数，因此不需要清零梯度
 * 但为了兼容接口，保留此函数
 */
void ReLU::zero_grad() {}