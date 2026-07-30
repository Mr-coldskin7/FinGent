"""
Tests for config.py — ModelConfig, get/update/reset, presets.
"""

import pytest
from config import (
    ModelConfig,
    get_model_config,
    update_model_config,
    reset_model_config,
    MODEL_PRESETS,
    get_settings,
    _settings,
)
import config as config_module


@pytest.fixture(autouse=True)
def clean_runtime_config():
    """确保每个测试前后运行时配置都是干净的"""
    config_module._runtime_model_config = None
    yield
    config_module._runtime_model_config = None


class TestModelConfig:
    def test_default_values(self):
        mc = ModelConfig()
        assert mc.api_key == ""
        assert mc.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert mc.model_name == "qwen-max"
        assert mc.temperature == 0.5

    def test_custom_values(self):
        mc = ModelConfig(
            api_key="sk-test1234",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
            temperature=0.3,
        )
        assert mc.api_key == "sk-test1234"
        assert mc.model_name == "gpt-4o"

    def test_masked_api_key_normal(self):
        mc = ModelConfig(api_key="sk-1234567890abcdef")
        masked = mc.masked_api_key()
        assert masked.startswith("sk-")
        assert masked.endswith("cdef")
        assert "***" in masked
        # 原始 key 不应出现在 masked 中
        assert "1234567890" not in masked

    def test_masked_api_key_short(self):
        mc = ModelConfig(api_key="short")
        assert mc.masked_api_key() == "***"

    def test_masked_api_key_empty(self):
        mc = ModelConfig(api_key="")
        assert mc.masked_api_key() == "***"

    def test_to_dict_masked(self):
        mc = ModelConfig(api_key="sk-1234567890", model_name="test")
        d = mc.to_dict(mask_key=True)
        assert d["api_key"] != "sk-1234567890"
        assert "***" in d["api_key"]
        assert d["model_name"] == "test"

    def test_to_dict_unmasked(self):
        mc = ModelConfig(api_key="sk-1234567890", model_name="test")
        d = mc.to_dict(mask_key=False)
        assert d["api_key"] == "sk-1234567890"


class TestGetModelConfig:
    def test_returns_settings_default_when_no_runtime(self):
        config_module._runtime_model_config = None
        mc = get_model_config()
        s = get_settings()
        assert mc.api_key == s.qianwen_api_key
        assert mc.base_url == s.model_base_url
        assert mc.model_name == s.model_name

    def test_returns_runtime_when_set(self):
        runtime = ModelConfig(api_key="runtime-key", model_name="runtime-model")
        config_module._runtime_model_config = runtime
        mc = get_model_config()
        assert mc.api_key == "runtime-key"
        assert mc.model_name == "runtime-model"


class TestUpdateModelConfig:
    def test_update_sets_runtime(self):
        new_config = ModelConfig(api_key="new-key", model_name="new-model")
        result = update_model_config(new_config)
        assert result.api_key == "new-key"
        assert get_model_config().api_key == "new-key"

    def test_update_returns_config(self):
        new_config = ModelConfig(api_key="k", model_name="m")
        result = update_model_config(new_config)
        assert isinstance(result, ModelConfig)


class TestResetModelConfig:
    def test_reset_clears_runtime(self):
        update_model_config(ModelConfig(api_key="temp"))
        reset_model_config()
        mc = get_model_config()
        s = get_settings()
        assert mc.api_key == s.qianwen_api_key

    def test_reset_returns_default(self):
        update_model_config(ModelConfig(api_key="temp"))
        result = reset_model_config()
        s = get_settings()
        assert result.api_key == s.qianwen_api_key


class TestModelPresets:
    def test_presets_exist(self):
        assert "DashScope (通义千问)" in MODEL_PRESETS
        assert "OpenAI" in MODEL_PRESETS
        assert "DeepSeek" in MODEL_PRESETS
        assert "Ollama (本地)" in MODEL_PRESETS

    def test_preset_has_required_fields(self):
        for name, preset in MODEL_PRESETS.items():
            assert "base_url" in preset, f"{name} missing base_url"
            assert "model_name" in preset, f"{name} missing model_name"
            assert preset["base_url"].startswith("http"), f"{name} invalid base_url"

    def test_dashscope_preset(self):
        p = MODEL_PRESETS["DashScope (通义千问)"]
        assert "dashscope" in p["base_url"]
        assert p["model_name"] == "qwen-max"

    def test_openai_preset(self):
        p = MODEL_PRESETS["OpenAI"]
        assert "openai.com" in p["base_url"]
        assert p["model_name"] == "gpt-4o"

    def test_deepseek_preset(self):
        p = MODEL_PRESETS["DeepSeek"]
        assert "deepseek" in p["base_url"]
        assert p["model_name"] == "deepseek-chat"
