"""
FinGent 统一配置管理
所有环境变量集中在这里，通过 Settings 单例访问。
支持运行时动态切换模型配置（ModelConfig）。
"""

import os
from typing import Optional
from dataclasses import dataclass, field, asdict
from pydantic_settings import BaseSettings
from pydantic import Field


@dataclass
class ModelConfig:
    """LLM 模型配置 — 运行时可动态修改"""
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-max"
    temperature: float = 0.5

    def masked_api_key(self) -> str:
        """返回脱敏的 API Key（如 sk-***1234）"""
        if not self.api_key or len(self.api_key) < 8:
            return "***"
        return f"{self.api_key[:3]}***{self.api_key[-4:]}"

    def to_dict(self, mask_key: bool = True) -> dict:
        """转为字典，可选脱敏 api_key"""
        d = asdict(self)
        if mask_key:
            d["api_key"] = self.masked_api_key()
        return d


# ── 预设提供商 ────────────────────────────────────────────────────────
MODEL_PRESETS: dict[str, dict] = {
    "DashScope (通义千问)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
    },
    "Ollama (本地)": {
        "base_url": "http://localhost:11434/v1",
        "model_name": "qwen2.5:7b",
    },
    "Moonshot (月之暗面)": {
        "base_url": "https://api.moonshot.cn/v1",
        "model_name": "moonshot-v1-128k",
    },
}


class Settings(BaseSettings):
    """FinGent 全局配置 — 从 .env 文件和环境变量自动加载"""

    # ── LLM ──────────────────────────────────────────────────────────
    qianwen_api_key: str = Field(default="", alias="QIANWEN_API_KEY")
    model_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="MODEL_BASE_URL",
    )
    model_name: str = Field(default="qwen-max", alias="MODEL_NAME")
    model_temperature: float = Field(default=0.5, alias="MODEL_TEMPERATURE")

    # ── 数据源 API Keys ──────────────────────────────────────────────
    fmp_api_key: str = Field(default="", alias="FMP_API_KEY")
    tiingo_api_key: str = Field(default="", alias="TIINGO_API_KEY")
    bocha_api_key: str = Field(default="", alias="BOCHA_API_KEY")

    # ── 存储 ─────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    mysql_url: str = Field(
        default="mysql+pymysql://root:password@localhost:3306/fingent",
        alias="MYSQL_URL",
    )
    memory_db_path: Optional[str] = Field(default=None, alias="FINGENT_MEMORY_DB")

    # ── 缓存 ─────────────────────────────────────────────────────────
    cache_default_ttl: int = Field(default=300, alias="CACHE_DEFAULT_TTL")
    cache_redis_ttl: int = Field(default=600, alias="CACHE_REDIS_TTL")

    # ── 投票阈值 ─────────────────────────────────────────────────────
    vote_strong_threshold: float = Field(default=0.75, alias="VOTE_STRONG_THRESHOLD")
    vote_majority_threshold: float = Field(default=0.4, alias="VOTE_MAJORITY_THRESHOLD")
    vote_reduce_threshold: float = Field(default=0.4, alias="VOTE_REDUCE_THRESHOLD")

    # ── 服务器 ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── 回测 ─────────────────────────────────────────────────────────
    backtest_max_concurrent: int = Field(default=5, alias="BACKTEST_MAX_CONCURRENT")
    backtest_timeout: int = Field(default=7200, alias="BACKTEST_TIMEOUT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def cors_origins_list(self) -> list[str]:
        """将逗号分隔的 CORS origins 字符串转为列表"""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",")]


# 全局单例
_settings: Optional[Settings] = None
_runtime_model_config: Optional[ModelConfig] = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_model_config() -> ModelConfig:
    """获取当前模型配置（运行时 > Settings 默认值）"""
    global _runtime_model_config
    if _runtime_model_config is not None:
        return _runtime_model_config
    s = get_settings()
    return ModelConfig(
        api_key=s.qianwen_api_key,
        base_url=s.model_base_url,
        model_name=s.model_name,
        temperature=s.model_temperature,
    )


def update_model_config(config: ModelConfig) -> ModelConfig:
    """更新运行时模型配置"""
    global _runtime_model_config
    _runtime_model_config = config
    return _runtime_model_config


def reset_model_config() -> ModelConfig:
    """重置为 .env 默认配置"""
    global _runtime_model_config
    _runtime_model_config = None
    return get_model_config()
