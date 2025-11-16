#pragma once

/**
 * @brief Loss基类
 * 
 * 所有损失函数的基类，定义了损失函数的接口
 */
class Loss {
public:
    /**
     * @brief 构造函数
     */
    Loss() = default;
    
    /**
     * @brief 虚析构函数
     */
    virtual ~Loss() = default;
    
    /**
     * @brief 前向传播接口，计算损失值
     * @param logits 模型输出的logits [batch_size, num_classes]
     * @param targets 真实标签 [batch_size]
     * @param batch_size 批次大小
     * @param num_classes 类别数量
     * @return 损失值
     */
    virtual float forward(const float* logits, const int* targets, int batch_size, int num_classes) = 0;
    
    /**
     * @brief 反向传播接口，计算梯度
     * @param d_logits logits的梯度 [batch_size, num_classes]
     * @param logits 模型输出的logits [batch_size, num_classes]
     * @param targets 真实标签 [batch_size]
     * @param batch_size 批次大小
     * @param num_classes 类别数量
     */
    virtual void backward(float* d_logits, const float* logits, const int* targets, int batch_size, int num_classes) = 0;
};