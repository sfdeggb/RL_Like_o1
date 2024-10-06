import argparse
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM
import os 

from train_rl import stage1_training_initialzation,stage2_training_with_reward_shaping
from utils import load_config


def main(config_file = 'config.yaml'):
    #读取配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)
    
    config = load_config(config_path)
    
    model_name = config['model_name'] 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 attn_implementation='eager')
    
    #load dataset
    data_file_path = config["data_file"]
    data_file_path_full = os.path.dirname(os.path.abspath(__file__))+data_file_path
    df = pd.read_csv(data_file_path_full,encoding='GB2312') 
    
    #prepare dataset
    data_stage1 =df[['question','original_answer']].to_dict(orient='records') 
    data_stage2 = df[['question','original_answer','correct_answer']].to_dict(orient='records')
    
    #STAGE 1: training (Intialzation)
    stage1_training_initialzation(model,tokenizer,data_stage1,config)
    
    #STAGE 2: training (Reward Shaping)
    stage2_training_with_reward_shaping(model,tokenizer,data_stage2,config)
    
    #save model
    model.save_pretrained(config['output_model_path'])
    tokenizer.save_pretrained(config['output_model_path'])
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file',type=str,default='config.yaml')
    # args = parser.parse_args()
    # main(args.config_file)
    main("config.yaml")
    
    
    
