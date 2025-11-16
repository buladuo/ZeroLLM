#pragma once
#include "loss.hpp"

/**
 * @brief Cross Entropy Loss类
 * 
 * 实现交叉熵损失函数，通常与softmax一起使用
 * 公式: loss = -log(softmax(logits)[i][targets[i]])
 */
class CrossEntropyLoss : public Loss {
private:
    float* softmax_output_;       // softmax输出缓存 [batch_size, num_classes]
    int last_batch_size_;         // 上一次前向传播的批次大小
    int last_num_classes_;        // 上一次前向传播的类别数量
    float eps_;                   // 数值稳定性参数

public:
    /**
     * @brief CrossEntropyLoss构造函数
     * @param eps 数值稳定性参数，默认为1e-8
     */
    explicit CrossEntropyLoss(float eps = 1e-8f);
    
    /**
     * @brief CrossEntropyLoss析构函数
     */
    ~CrossEntropyLoss();
    
    /**
     * @brief 前向传播，计算交叉熵损失
     * @param logits 模型输出的logits [batch_size, num_classes]
     * @param targets 真实标签 [batch_size]
     * @param batch_size 批次大小
     * @param num_classes 类别数量
     * @return 平均损失值
     */
    float forward(const float* logits, const int* targets, int batch_size, int num_classes) override;
    
    /**
     * @brief 反向传播，计算梯度
     * @param d_logits logits的梯度 [batch_size, num_classes]
     * @param logits 模型输出的logits [batch_size, num_classes]
     * @param targets 真实标签 [batch_size]
     * @param batch_size 批次大小
     * @param num_classes 类别数量
     */
    void backward(float* d_logits, const float* logits, const int* targets, int batch_size, int num_classes) override;
    
    /**
     * @brief 获取softmax输出缓存
     * @return softmax输出缓存指针
     */
    const float* softmax_output() const { return softmax_output_; }
};