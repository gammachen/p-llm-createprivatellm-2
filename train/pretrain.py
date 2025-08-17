import os
import json
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tokenizer(config):
    """
    创建并配置BERT分词器
    
    该函数根据配置加载预训练的BERT分词器，并对其进行特定的token配置，
    包括设置开始和结束token，以及处理填充token的缺失情况。
    
    参数:
        config (dict): 配置字典，必须包含"tokenizer_name"键，用于指定预训练分词器的名称
    
    返回:
        BertTokenizerFast: 配置完成的BERT快速分词器实例
    """
    # 从预训练模型加载BERT分词器
    tokenizer = BertTokenizerFast.from_pretrained(config["tokenizer_name"])
    
    # 将BERT的特殊token映射为通用的开始和结束token
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    
    # 如果分词器没有设置填充token，则使用结束token作为填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_model(tokenizer, config):
    """
    创建并配置GPT-2语言模型
    
    参数:
        tokenizer: 分词器对象，用于获取词汇表大小和特殊token ID
        config: 配置字典，包含模型名称、最大长度等配置参数
    
    返回:
        配置好的GPT-2语言模型对象
    """
    # 获取模型名称并确定运行设备
    model_name = config["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading {model_name} to device: {device}")
    
    # 从预训练模型加载配置并根据tokenizer调整参数
    model_config = GPT2Config.from_pretrained(
        model_name,
        vocab_size=tokenizer.vocab_size,
        n_positions=config["max_len"],
        n_ctx=config["max_len"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 创建GPT-2语言模型实例并调整词嵌入层大小
    model = GPT2LMHeadModel(model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model


def create_dataset(tokenizer, config):
    """
    创建并预处理训练和评估数据集
    
    该函数从wiki40b数据集中加载中文数据，将其分割为训练集和测试集，
    然后使用指定的tokenizer对文本进行分词和预处理。
    
    参数:
        tokenizer: 用于文本分词的tokenizer对象
        config: 配置字典，必须包含"max_len"键指定最大序列长度
    
    返回:
        tuple: 包含两个元素的元组
            - tokenized_train_dataset: 预处理后的训练数据集
            - tokenized_eval_dataset: 预处理后的评估数据集
    """
    def tokenize_function(sample):
        # 对单个样本进行tokenization处理，包括截断和填充
        return tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=config["max_len"]
        )

    # 加载wiki40b中文数据集的训练分割
    dataset = load_dataset("wiki40b", "zh-cn", split="train")
    # 将数据集按9:1的比例分割为训练集和测试集
    dataset = dataset.train_test_split(test_size=0.1)
    # 获取并打乱训练数据集，可选地限制样本数量用于测试
    train_dataset = dataset["train"].shuffle()
    # 获取并打乱测试数据集，可选地限制样本数量用于测试
    eval_dataset = dataset["test"].shuffle()
    
    # 定义要移除的列
    columns_to_remove = ["text", "wikidata_id", "version_id"]
    
    # 对训练数据集进行批量tokenization处理，移除原始文本列和其他无关列
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=columns_to_remove
    )
    
    # 对评估数据集进行批量tokenization处理，移除原始文本列和其他无关列
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=columns_to_remove
    )
    
    return tokenized_train_dataset, tokenized_eval_dataset


def create_training_arguments(config):
    """
    创建训练参数配置对象
    
    该函数根据传入的配置字典创建一个TrainingArguments对象，用于配置模型训练的各种参数，
    包括输出路径、评估策略、学习率、批次大小等训练相关设置。
    
    参数:
        config (dict): 配置字典，必须包含"model_path"键，指定模型输出目录路径
        
    返回:
        TrainingArguments: 配置好的训练参数对象，用于模型训练
    """
    training_args = TrainingArguments(
        output_dir=config["model_path"],
        evaluation_strategy="steps",  # 使用evaluation_strategy替代已弃用的eval_strategy
        eval_steps=1000,
        save_steps=1000,
        logging_steps=1000,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        fp16=False,
        fp16_full_eval=False,
        dataloader_num_workers=2,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        report_to=[],
    )
    return training_args


def create_trainer(model, training_args, data_collator, train_data, eval_data, tokenizer):
    """
    创建一个用于模型训练的Trainer实例
    
    参数:
        model: 要训练的模型对象
        training_args: 训练参数配置对象，包含训练过程的各种超参数
        data_collator: 数据整理器，用于将样本整理成批次数据
        train_data: 训练数据集
        eval_data: 验证数据集
        tokenizer: 分词器，用于文本预处理
    
    返回:
        Trainer: 配置好的训练器实例
    """
    # 创建Trainer实例，用于管理模型的训练和评估过程
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
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


def pretrain():
    """
    预训练模型的主函数
    
    该函数负责执行完整的预训练流程，包括：
    1. 解析命令行参数获取配置文件路径
    2. 加载配置文件并初始化训练环境
    3. 创建分词器、模型和训练相关组件
    4. 执行模型训练并保存训练结果
    
    参数:
        无直接参数，通过命令行传入
        
    命令行参数:
        --config (str): 必需参数，指定配置文件的路径，例如: config/pretrain.json
        
    返回值:
        无返回值
    """
    parser = argparse.ArgumentParser(description="Pretrain config")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指定配置文件的路径，例如: config/pretrain.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger.info("Config:\n%s", json.dumps(config, indent=4, ensure_ascii=False))

    # 初始化分词器和模型
    tokenizer = create_tokenizer(config)
    model = create_model(tokenizer, config)
    
    # 创建训练参数和数据处理组件
    training_args = create_training_arguments(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 准备训练和验证数据集
    train_data, eval_data = create_dataset(tokenizer, config)
    
    # 创建训练器并执行训练
    trainer = create_trainer(model, training_args, data_collator, train_data, eval_data, tokenizer)
    trainer.train()
    
    # 保存训练好的模型和分词器
    trainer.save_model(config["model_path"])
    tokenizer.save_pretrained(config["model_path"])


if __name__ == '__main__':
    pretrain()