#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <algorithm>

#include "bpe_tokenizer.hpp"
#include "recorder.hpp"
#include "dataloader.hpp"
#include "zerollm.hpp"
#include "zerollm_config.hpp"
#include "optimizer.hpp"
#include "adamw.hpp"
#include "config.hpp"
#include "loss.hpp"
#include "cross_entropy.hpp"
#include "async_logger.hpp"


int main(){
    auto model_config = create_zerollm_26m_config();
    ZeroLLM model = ZeroLLM(model_config);
    // ------------------------------------------------------
    // 操作 1: 获取扁平化参数列表
    // ------------------------------------------------------
    std::vector<Parameter> all_params;
    model.get_parameters(all_params, "zerollm");
    std::cout << "[Test 1] Total Parameter Objects found: " << all_params.size() << std::endl;
    
    // ------------------------------------------------------
    // 操作 2: 打印结构 (Verify Hierarchy)
    // ------------------------------------------------------
    std::cout << "\n[Test 2] Parameter Hierarchy:" << std::endl;
    size_t total_elements = 0;
    for (const auto& p : all_params) {
        std::cout << "  Name: " << p.name 
                  << "\t| Size: " << p.size 
                  << "\t| Address: " << p.data << std::endl;
        total_elements += p.size;
    }
    std::cout << "  >> Total Param Count: " << total_elements << std::endl;

    model.print_structure();
}