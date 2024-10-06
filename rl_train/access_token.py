import os 
from huggingface_hub import login
from utils import load_config

config_path = os.path.dirname(os.path.abspath(__file__))+"\\config.yaml"

config = load_config(config_path)

login(token=config["access_token"])