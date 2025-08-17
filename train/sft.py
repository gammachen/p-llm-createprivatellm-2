import os
import json
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def formatting_prompt(sample):
    """
    格式化提示词
    
    参数:
        sample (dict): 包含指令、输入和响应的数据样本
        
    返回:
        str: 格式化后的提示词字符串
    """
    instruction = sample["INSTRUCTION"]
    input_text = sample.get("INPUT", "")
    response = sample["RESPONSE"]
    prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n{response}\n"
    return prompt


def create_dataset(config):
    """
    创建训练数据集
    
    参数:
        config (dict): 配置字典
        
    返回:
        Dataset: 分割后的数据集，包含train和test两个分割
    """
    dataset = load_dataset("wangrui6/Zhihu-KOL", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


def create_training_arguments(config):
    """
    创建训练参数
    
    参数:
        config (dict): 配置字典
        
    返回:
        SFTConfig: 训练配置对象
    """
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        evaluation_strategy="epoch",  # 使用evaluation_strategy替代已弃用的eval_strategy
        save_strategy="steps",
        save_steps=5000,
        logging_steps=2000,
        learning_rate=5e-5,
        save_total_limit=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to=[],
        max_seq_length=512
    )
    return training_args


def create_trainer(model, training_args, data_collator, dataset, formatting_func):
    """
    创建训练器
    
    参数:
        model: 模型对象
        training_args: 训练参数
        data_collator: 数据整理器
        dataset: 数据集
        formatting_func: 格式化函数
        
    返回:
        SFTTrainer: 训练器实例
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_func,
    )
    return trainer


def load_config(config_path):
    """
    加载配置文件
    
    参数:
        config_path (str): 配置文件路径
        
    返回:
        dict: 配置字典
        
    异常:
        FileNotFoundError: 当配置文件不存在时抛出
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def sft():
    """
    SFT训练主函数
    """
    parser = argparse.ArgumentParser(description="SFT config")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指定配置文件的路径，例如: config/sft.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger.info("Config:\n%s", json.dumps(config, indent=4, ensure_ascii=False))

    model_path = config["model_path"]
    output_dir = config["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    training_args = create_training_arguments(config)
    response_template = "### Response:\n"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
    dataset = create_dataset(config)
    trainer = create_trainer(model, training_args, data_collator, dataset, formatting_prompt)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    sft()