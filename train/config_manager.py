import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """基础配置类"""
    model_path: str
    output_dir: str
    max_len: int = 512
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    fp16: bool = False
    fp16_full_eval: bool = False
    dataloader_num_workers: int = 2
    report_to: list = field(default_factory=list)
    
    def __post_init__(self):
        """配置验证"""
        self.validate()
    
    def validate(self):
        """验证配置参数"""
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("训练批次大小必须大于0")
        if self.num_train_epochs <= 0:
            raise ValueError("训练轮数必须大于0")
        if self.max_len <= 0:
            raise ValueError("最大序列长度必须大于0")


@dataclass
class PretrainConfig(BaseConfig):
    """预训练配置"""
    model_name: str = "gpt2"
    tokenizer_name: str = "bert-base-chinese"
    
    def validate(self):
        super().validate()
        if not self.model_name:
            raise ValueError("模型名称不能为空")
        if not self.tokenizer_name:
            raise ValueError("分词器名称不能为空")


@dataclass
class SFTConfig(BaseConfig):
    """SFT训练配置"""
    max_seq_length: int = 512
    
    def validate(self):
        super().validate()
        if self.max_seq_length <= 0:
            raise ValueError("最大序列长度必须大于0")


@dataclass
class DPOConfig(BaseConfig):
    """DPO训练配置"""
    max_length: int = 512
    
    def validate(self):
        super().validate()
        if self.max_length <= 0:
            raise ValueError("最大长度必须大于0")


@dataclass
class LoRAConfig(BaseConfig):
    """LoRA训练配置"""
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["lm_head"])
    
    def validate(self):
        super().validate()
        if self.lora_r <= 0:
            raise ValueError("LoRA rank必须大于0")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha必须大于0")
        if not (0 <= self.lora_dropout <= 1):
            raise ValueError("LoRA dropout必须在0-1之间")


class ConfigManager:
    """配置管理器"""
    
    CONFIG_CLASSES = {
        'pretrain': PretrainConfig,
        'sft': SFTConfig,
        'dpo': DPOConfig,
        'lora': LoRAConfig
    }
    
    @staticmethod
    def load_config(config_path: str, config_type: str) -> BaseConfig:
        """
        加载并验证配置文件
        
        参数:
            config_path: 配置文件路径
            config_type: 配置类型 ('pretrain', 'sft', 'dpo', 'lora')
            
        返回:
            配置对象
            
        异常:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置类型不支持或配置参数无效
            json.JSONDecodeError: JSON格式错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
        
        if config_type not in ConfigManager.CONFIG_CLASSES:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"配置文件JSON格式错误: {e}", e.doc, e.pos)
        
        config_class = ConfigManager.CONFIG_CLASSES[config_type]
        
        try:
            config = config_class(**config_dict)
            logger.info(f"成功加载{config_type}配置: {config_path}")
            return config
        except TypeError as e:
            raise ValueError(f"配置参数错误: {e}")
    
    @staticmethod
    def save_config(config: BaseConfig, config_path: str) -> None:
        """
        保存配置到文件
        
        参数:
            config: 配置对象
            config_path: 保存路径
        """
        # 确保目录存在
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典并保存
        config_dict = config.__dict__.copy()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {config_path}")
    
    @staticmethod
    def create_default_config(config_type: str, **kwargs) -> BaseConfig:
        """
        创建默认配置
        
        参数:
            config_type: 配置类型
            **kwargs: 覆盖的配置参数
            
        返回:
            配置对象
        """
        if config_type not in ConfigManager.CONFIG_CLASSES:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        config_class = ConfigManager.CONFIG_CLASSES[config_type]
        
        # 设置必需的默认值
        defaults = {
            'model_path': './models/base_model',
            'output_dir': f'./output/{config_type}'
        }
        defaults.update(kwargs)
        
        return config_class(**defaults)