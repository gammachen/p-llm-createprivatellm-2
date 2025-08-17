import os
import json
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def formatting(sample):
    """
    格式化样本数据
    
    参数:
        sample (dict): 包含prompt、chosen和rejected键的样本数据
        
    返回:
        dict: 格式化后的数据
    """
    prompt = f"### Instruction:\n{sample['prompt']}### Input:\n\n### Response:\n"
    chosen = f"{sample['chosen']}\n"
    rejected = f"{sample['rejected']}\n"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def create_dataset(config):
    """
    创建训练数据集
    
    参数:
        config (dict): 配置字典
        
    返回:
        Dataset: 分割后的数据集，包含train和test两个分割
    """
    dataset = load_dataset("liyucheng/zhihu_rlhf_3k", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


def create_training_arguments(config):
    """
    创建训练参数
    
    参数:
        config (dict): 配置字典
        
    返回:
        DPOConfig: 训练配置对象
    """
    training_args = DPOConfig(
        output_dir=config["output_dir"],
        evaluation_strategy="epoch",  # 使用evaluation_strategy替代已弃用的eval_strategy
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1000,
        learning_rate=5e-5,
        save_total_limit=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to=[],
        max_length=512
    )
    return training_args


def create_trainer(model, ref_model, training_args, dataset, tokenizer):
    """
    创建DPO训练器
    
    参数:
        model: 主模型
        ref_model: 参考模型
        training_args: 训练参数
        dataset: 数据集
        tokenizer: 分词器
        
    返回:
        DPOTrainer: 配置好的DPO训练器实例
    """
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
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


def dpo():
    """
    DPO训练主函数
    """
    parser = argparse.ArgumentParser(description="DPO config")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指定配置文件的路径，例如: config/dpo.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger.info("Config:\n%s", json.dumps(config, indent=4, ensure_ascii=False))

    model_path = config["model_path"]
    output_dir = config["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    training_args = create_training_arguments(config)
    dataset = create_dataset(config)
    trainer = create_trainer(model, ref_model, training_args, dataset, tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    dpo()