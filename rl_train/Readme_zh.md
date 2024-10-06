# 🧠 自校正模型与SCoRe数据集

本项目实现了一个**自校正模型**，该模型在**SCoRe数据集**上对一个大规模语言模型进行微调。模型设计用于生成初始答案，自我校正，并使用强化学习技术改进其响应。

## 📚 项目概述

- **模型**：该模型基于`gemma-2-2B-it`架构，在SCoRe数据集上进行微调。
- **训练**：模型经历两个阶段的训练过程。在第一阶段，模型生成初始答案，在第二阶段，模型根据奖励系统自我校正其答案。
- **强化学习**：强化学习用于通过奖励更好的答案并惩罚不正确或未更改的响应来改进自我校正过程。

### 🛠 关键特性 

1. **自校正**：模型生成初始答案（`y1`），然后根据反馈提供一个校正后的答案（`y2`）。如果校正后的答案改进了原始答案，则给予奖励。
2. **SCoRe数据集**：模型在**SCoRe数据集**上进行训练，该数据集包含问题、初始答案和正确答案，允许模型从成功和失败中学习。
3. **聊天模板**：使用`chat_template`结构化用户和模型之间的交互，便于对模型进行对话AI任务的微调。

## 🚀 工作原理

### 1. **阶段I：初始答案生成**
   - 模型根据输入问题生成初始答案（`y1`）。
   - 结合**交叉熵损失**和**KL散度损失**，引导模型在原始分布附近生成合理的答案。

### 2. **阶段II：自我校正与奖励塑造**
   - 模型在接收到反馈后，生成第二次尝试（`y2`）。
   - **奖励函数**：
     - 如果第二次答案改进了原始答案，则给予奖励。
     - 如果没有改进，则给予惩罚。
     - 部分改进也会给予奖励，但奖励较低。
   - **塑造损失**是通过调整损失，根据模型在第二阶段获得的奖励来计算的。

## 📦 数据集

SCoRe数据集包含以下字段：
- `question`: 问题文本。
- `original_answer`: 初始答案。
- `correct_answer`: 正确答案。

### 示例
| question                       | original_answer         | correct_answer                |
|---------------------------------|-------------------------|-------------------------------|
| 2 + 2是什么？                  | 2 + 2 = 3               | 2 + 2 = 4                     |
| 法国的首都是哪里？            | 法国是一个国家          | 法国的首都是巴黎            |

## 🛠 如何运行模型  

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **运行训练脚本**：
   ```bash
   python train.py --config config.yaml
   ```      

## 结果
* 错误答案
<img src="./image/gemma-2_wrong.png" alt="Wrong Predict" width="500"/>

* 正确答案
<img src="./image/gemma-2_tuned_answer.png" alt="Right Predict" width="500"/>