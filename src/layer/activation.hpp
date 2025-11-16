#pragma once

/**
 * @brief 激活函数基类
 * 
 * 所有激活函数类的基类，定义了激活函数的接口
 */
class Activation {
protected:
    const float* input_;            // 输入数据指针
    float* output_;                 // 输出数据指针
    int last_batch_size_;           // 上一次前向传播的批次大小
    int features_;                  // 特征数

public:
    /**
     * @brief 激活函数基类构造函数
     */
    Activation() : input_(nullptr), output_(nullptr), last_batch_size_(0), features_(0) {}
    
    /**
     * @brief 虚析构函数
     */
    virtual ~Activation() = default;
    
    /**
     * @brief 前向传播接口
     * @param output 输出数据指针
     * @param input 输入数据指针
     * @param batch_size 批次大小
     * @param features 特征数
     */
    virtual void forward(float* output, const float* input, int batch_size, int features) {
        this->input_ = input;
        this->output_ = output;
        this->last_batch_size_ = batch_size;
        this->features_ = features;
    }
    
    /**
     * @brief 反向传播接口
     * @param d_input 输入梯度
     * @param d_output 输出梯度
     */
    virtual void backward(float* d_input, const float* d_output) = 0;
};