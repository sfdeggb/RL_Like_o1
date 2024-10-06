import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
import bert_score
import yaml 
from utils import stage1_chat_template,stage2_chat_template,chat_format



#reward function for self-correction 
def reward_function(original_answer, corrected_answer, correct_answer):
    if correct_answer == corrected_answer:
        return 1
    elif original_answer == corrected_answer:
        return -1
    else:
        return is_improved(original_answer, corrected_answer, correct_answer)

def is_improved(original_answer: str, corrected_answer: str, correct_answer: str, language: str = 'zh') -> float:
    P_pos, R_pos, F1_pos = bert_score.score(corrected_answer, correct_answer, 
                                    lang=language, 
									verbose=True,
									model_type='bert-large-uncased') 
    P_neg, R_neg, F1_neg = bert_score.score(corrected_answer, original_answer, 
                                lang=language, 
                                verbose=True,
                                model_type='bert-large-uncased') 
    improve_degree: float = F1_pos.mean().item() - F1_neg.mean().item()
    return improve_degree


# STAGE1: Training initial model to genetate frist attempt(y1) and prevent mode collapse 
def stage1_training_initialzation(model,tokenizer,data,config):
    optimizer = optimizers.AdamW(model.parameters(),lr=float(config['learning_rate']))
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    
    for epoch in range(config['epochs_stage_1']):
        total_loss = 0.0
        for example in data:
            conversation =stage1_chat_template(example)
            #convert conversation to singal string
            conversation_text = tokenizer.apply_chat_template(conversation,tokenize=False)
            
            inputs = tokenizer(conversation_text,return_tensors='pt',
                               padding=True,truncation=True)
            
            inputs = {k:v.to(model.device) for k,v in inputs.items()}
            
            outputs = model(**inputs,labels=inputs['input_ids'])
            #cross entropy loss
            cross_entropy_loss = outputs.loss
            # 从模型输出中获取logits
            # logits是模型的原始输出，表示每个词的预测分数
            # 这些分数还未经过softmax转换为概率
            logits = outputs.logits
            log_probs = F.log_softmax(logits,dim=-1)
            
            with torch.no_grad():
                target_probs = F.softmax(logits,dim=-1)
            # 计算KL散度损失
            # F.kl_div计算KL散度，reduction='batchmean'表示计算每个样本的KL散度并取平均
            kl_loss = F.kl_div(log_probs,target_probs,reduction='batchmean')
            #total loss combines cross entropy loss and kl divergence loss
            total_loss_value = cross_entropy_loss + config['beta_kl']*kl_loss
            
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()
            
            total_loss += total_loss_value.item()
        print(f"Stage 1: Epoch {epoch+1}/{config['epochs_stage_1']}, Loss: {total_loss:.4f}")


#STAGE2: Training model to generate high quality responses
def stage2_training_with_reward_shaping(model,tokenizer,data,config):
    optimizer = optimizers.AdamW(model.parameters(),lr=config['learning_rate'])
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(config['epochs_stage_2']):
        total_loss = 0.0
        for example in data:
            # Frist attempt(y1):Generate the initial answer using chat template 
            conversation1= stage2_chat_template(example)
            conversation_text1 = tokenizer.apply_chat_template(conversation1,tokenize=False)
            
            inputs1 = tokenizer(conversation_text1,return_tensors='pt',padding=True,truncation=True).to(device)
            
            inputs1 = {k:v.to(model.device) for k,v in inputs1.items()}           
            # torch.no_grad()用于暂时禁用梯度计算，以减少内存使用并加快计算速度。
            # 在推理阶段或不需要计算梯度的操作中使用它可以提高效率。
            with torch.no_grad():
                outputs1 = model(**inputs1)
            #Second attempt(y2):corrected answer
            conversation2= chat_format(example)
            conversation_text2 = tokenizer.apply_chat_template(conversation2,tokenize=False)
            inputs2 = tokenizer(conversation_text2,return_tensors='pt',padding=True,truncation=True).to(device)
            inputs2 = {k:v.to(model.device) for k,v in inputs2.items()}
            # 使用模型生成第二个回答
            outputs2 = model(**inputs2,labels=inputs2['input_ids'])
            
            #Ensure we have a loss 
            if outputs2.loss is None:
                print(f"Warning: No loss for outputs2. using cross entropy loss instead.")
                logits = outputs2.logits
                loss = F.cross_entropy(logits.view(-1,logits.size(-1)),
                                       inputs2['input_ids'].view(-1))
            else:
                loss = outputs2.loss
            # calculate reward based on self-correction
            generated_text1 = tokenizer.decode(outputs1.logits.argmax(dim=-1)[0],skip_special_tokens=True)
            generated_text2 = tokenizer.decode(outputs2.logits.argmax(dim=-1)[0],skip_special_tokens=True)
            
            reward = reward_function(example['original_answer'],generated_text2,example['correct_answer'])
            
            #apply reward shaping
            shaped_loss = loss*reward
            
            optimizer.zero_grad()
            shaped_loss.backward()
            optimizer.step()
            
            total_loss += shaped_loss.item()
        print(f"Stage 2: Epoch {epoch+1}/{config['epochs_stage_2']}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    print(is_improved("2+2等于5","2+2等于3","2+2等于4"))



    
