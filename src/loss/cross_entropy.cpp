#include "cross_entropy.hpp"
#include "cross_entropy.cuh"
#include "config.hpp"
#include "softmax.cuh"
#include "utils/async_logger.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

CrossEntropyLoss::CrossEntropyLoss(float eps)
    : softmax_output_(nullptr),
      last_batch_size_(0),
      last_num_classes_(0),
      eps_(eps) {
}

CrossEntropyLoss::~CrossEntropyLoss() {
    if (softmax_output_) {
        zerollm_backend::free(softmax_output_);
    }
}

float CrossEntropyLoss::forward(const float* logits, const int* targets, int batch_size, int num_classes) {
    LOG_DEBUG("CrossEntropyLoss forward: batch_size=" << batch_size << ", num_classes=" << num_classes);
    
    if (batch_size <= 0 || num_classes <= 0) {
        throw std::invalid_argument("Batch size and number of classes must be positive");
    }
    
    // 重新分配缓冲区
    if (last_batch_size_ != batch_size || last_num_classes_ != num_classes) {
        LOG_DEBUG("Resizing softmax output buffer");
        if (softmax_output_) {
            zerollm_backend::free(softmax_output_);
        }
        
        softmax_output_ = (float*)zerollm_backend::malloc(batch_size * num_classes * sizeof(float));
        last_batch_size_ = batch_size;
        last_num_classes_ = num_classes;
    }
    
    // 计算softmax
    LOG_DEBUG("Computing softmax");
    softmax(logits, softmax_output_, batch_size, num_classes, 0);
    zerollm_backend::device_synchronize();
    
    // 在设备上计算损失
    float* d_losses = (float*)zerollm_backend::malloc(batch_size * sizeof(float));
    
    // 创建CUDA内核来计算每个样本的损失
    float* h_losses = new float[batch_size];
    int* h_targets = new int[batch_size];
    
    zerollm_backend::memcpy(h_targets, targets, batch_size * sizeof(int), zerollm_backend::CopyKind::D2H);
    
    // 计算交叉熵: -log(softmax_output[target])
    float total_loss = 0.0f;
    float* h_softmax = new float[batch_size * num_classes];
    zerollm_backend::memcpy(h_softmax, softmax_output_, batch_size * num_classes * sizeof(float), zerollm_backend::CopyKind::D2H);
    
    for (int i = 0; i < batch_size; ++i) {
        int target = h_targets[i];
        if (target >= 0 && target < num_classes) {
            float prob = h_softmax[i * num_classes + target];
            // 添加数值稳定性保护
            prob = fmaxf(prob, eps_);
            h_losses[i] = -logf(prob);
            total_loss += h_losses[i];
        } else {
            h_losses[i] = 0.0f;
        }
    }
    
    delete[] h_softmax;
    delete[] h_targets;
    delete[] h_losses;
    zerollm_backend::free(d_losses);
    
    float avg_loss = total_loss / batch_size;
    LOG_DEBUG("CrossEntropyLoss forward completed. Average loss: " << avg_loss);
    return avg_loss;
}

void CrossEntropyLoss::backward(float* d_logits, const float* logits, const int* targets, int batch_size, int num_classes) {
    LOG_DEBUG("CrossEntropyLoss backward: batch_size=" << batch_size << ", num_classes=" << num_classes);
    
    if (!softmax_output_) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    // 使用CUDA内核计算梯度
    LOG_DEBUG("Computing gradients with CUDA kernel");
    cross_entropy_backward_kernel(d_logits, softmax_output_, targets, batch_size, num_classes, 0);
    zerollm_backend::device_synchronize();
    
    LOG_DEBUG("CrossEntropyLoss backward completed");
}