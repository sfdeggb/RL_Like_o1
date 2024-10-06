import yaml 

def stage1_chat_template(example):
    return [
        {
            'role':'user',
            'content':example['question']
        },
        {
            'role':'assistant',
            "content": f"答案: {example.get('original_answer', '')}"
        }
    ]
def stage2_chat_template(example):
    return [
        {
            'role':'user',
            'content':example['question']
        },
        {"role": "assistant", 
         "content": f"第一个答案: {example.get('original_answer', '')}"
        },
        {
            'role':'user',
            'content':"请纠正第一个答案中的错误，并给出正确的答案"
        },
        {
            'role':'assistant',
            'content':"研究后的答案 "
        }
    ]
def chat_format(example):
    return [
        {
            'role':'user',
            'content':example['question']
        },
        {"role": "assistant", 
         "content": f"最终答案: {example.get('correct_answer', '')}"}
    ]
#load configuration from a Yaml file
def load_config(config_file='config.yaml'):
    if config_file:
        with open(config_file,'r',encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config
