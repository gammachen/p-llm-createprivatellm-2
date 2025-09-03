import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Callable
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BaseDatasetManager(ABC):
    """数据集管理基类"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.get('max_len', 512)
    
    @abstractmethod
    def load_raw_dataset(self) -> Dataset:
        """加载原始数据集"""
        pass
    
    @abstractmethod
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """格式化单个样本"""
        pass
    
    def tokenize_function(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """默认的tokenization函数"""
        if isinstance(samples.get('text'), list):
            # 批量处理
            texts = samples['text']
        else:
            # 单个样本
            texts = [samples['text']]
        
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None
        )
    
    def create_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        创建训练和评估数据集
        
        返回:
            Tuple[Dataset, Dataset]: 训练数据集和评估数据集
        """
        try:
            # 加载原始数据集
            dataset = self.load_raw_dataset()
            logger.info(f"原始数据集大小: {len(dataset)}")
            
            # 分割数据集
            test_size = self.config.get('test_size', 0.1)
            dataset = dataset.train_test_split(test_size=test_size)
            
            # 打乱数据
            train_dataset = dataset["train"].shuffle(seed=42)
            eval_dataset = dataset["test"].shuffle(seed=42)
            
            # 限制数据集大小（用于调试）
            max_train_samples = self.config.get('max_train_samples')
            max_eval_samples = self.config.get('max_eval_samples')
            
            if max_train_samples:
                train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
            if max_eval_samples:
                eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
            
            # 应用格式化和tokenization
            train_dataset = self.process_dataset(train_dataset)
            eval_dataset = self.process_dataset(eval_dataset)
            
            logger.info(f"处理后训练集大小: {len(train_dataset)}")
            logger.info(f"处理后评估集大小: {len(eval_dataset)}")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.error(f"创建数据集时发生错误: {e}")
            raise
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        处理数据集：格式化 + tokenization
        
        参数:
            dataset: 原始数据集
            
        返回:
            Dataset: 处理后的数据集
        """
        # 格式化样本
        dataset = dataset.map(
            self.format_sample,
            desc="格式化样本",
            num_proc=self.config.get('num_proc', 1)
        )
        
        # 获取需要移除的列
        columns_to_remove = self.get_columns_to_remove(dataset)
        
        # Tokenization
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            desc="Tokenizing",
            remove_columns=columns_to_remove,
            num_proc=self.config.get('num_proc', 1)
        )
        
        return tokenized_dataset
    
    def get_columns_to_remove(self, dataset: Dataset) -> list:
        """获取需要移除的列"""
        # 默认移除所有非模型输入的列
        keep_columns = {'input_ids', 'attention_mask', 'labels'}
        return [col for col in dataset.column_names if col not in keep_columns]


class PretrainDatasetManager(BaseDatasetManager):
    """预训练数据集管理器"""
    
    def load_raw_dataset(self) -> Dataset:
        """加载wiki40b中文数据集"""
        dataset_name = self.config.get('dataset_name', 'wiki40b')
        dataset_config = self.config.get('dataset_config', 'zh-cn')
        
        logger.info(f"加载数据集: {dataset_name} ({dataset_config})")
        return load_dataset(dataset_name, dataset_config, split="train")
    
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """格式化预训练样本"""
        return {'text': sample['text']}
    
    def get_columns_to_remove(self, dataset: Dataset) -> list:
        """预训练需要移除的列"""
        return ["text", "wikidata_id", "version_id"]


class SFTDatasetManager(BaseDatasetManager):
    """SFT数据集管理器"""
    
    def load_raw_dataset(self) -> Dataset:
        """加载SFT数据集"""
        dataset_name = self.config.get('dataset_name', 'wangrui6/Zhihu-KOL')
        logger.info(f"加载SFT数据集: {dataset_name}")
        return load_dataset(dataset_name, split="train")
    
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """格式化SFT样本"""
        instruction = sample["INSTRUCTION"]
        input_text = sample.get("INPUT", "")
        response = sample["RESPONSE"]
        
        prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n{response}\n"
        return {'text': prompt}
    
    def get_columns_to_remove(self, dataset: Dataset) -> list:
        """SFT需要移除的列"""
        return ["INSTRUCTION", "INPUT", "RESPONSE", "text"]


class DPODatasetManager(BaseDatasetManager):
    """DPO数据集管理器"""
    
    def load_raw_dataset(self) -> Dataset:
        """加载DPO数据集"""
        dataset_name = self.config.get('dataset_name', 'liyucheng/zhihu_rlhf_3k')
        logger.info(f"加载DPO数据集: {dataset_name}")
        return load_dataset(dataset_name, split="train")
    
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """格式化DPO样本"""
        prompt = f"### Instruction:\n{sample['prompt']}### Input:\n\n### Response:\n"
        chosen = f"{sample['chosen']}\n"
        rejected = f"{sample['rejected']}\n"
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    
    def tokenize_function(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """DPO特殊的tokenization"""
        # DPO不需要标准的tokenization，返回原始格式
        return samples
    
    def get_columns_to_remove(self, dataset: Dataset) -> list:
        """DPO不移除列，因为需要保留prompt, chosen, rejected"""
        return []


class LoRADatasetManager(BaseDatasetManager):
    """LoRA数据集管理器"""
    
    def load_raw_dataset(self) -> Dataset:
        """加载LoRA数据集"""
        dataset_name = self.config.get('dataset_name', 'liyucheng/zhihu_rlhf_3k')
        logger.info(f"加载LoRA数据集: {dataset_name}")
        return load_dataset(dataset_name, split="train")
    
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """格式化LoRA样本"""
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        text = f"### Instruction:\n{prompt}\n### Input:\n\n### Response:\n{chosen}\n"
        return {'text': text}
    
    def get_columns_to_remove(self, dataset: Dataset) -> list:
        """LoRA需要移除的列"""
        return ['prompt', 'chosen', 'rejected', 'text']


class DatasetManagerFactory:
    """数据集管理器工厂"""
    
    MANAGERS = {
        'pretrain': PretrainDatasetManager,
        'sft': SFTDatasetManager,
        'dpo': DPODatasetManager,
        'lora': LoRADatasetManager
    }
    
    @staticmethod
    def create_manager(dataset_type: str, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]) -> BaseDatasetManager:
        """
        创建数据集管理器
        
        参数:
            dataset_type: 数据集类型 ('pretrain', 'sft', 'dpo', 'lora')
            tokenizer: 分词器
            config: 配置字典
            
        返回:
            BaseDatasetManager: 数据集管理器实例
        """
        if dataset_type not in DatasetManagerFactory.MANAGERS:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        manager_class = DatasetManagerFactory.MANAGERS[dataset_type]
        return manager_class(tokenizer, config)