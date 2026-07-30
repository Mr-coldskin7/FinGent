<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface ModelConfig {
  api_key: string;
  base_url: string;
  model_name: string;
  temperature: number;
}

interface Preset {
  base_url: string;
  model_name: string;
}

const config = reactive<ModelConfig>({
  api_key: '',
  base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  model_name: 'qwen-max',
  temperature: 0.5,
});

const presets = ref<Record<string, Preset>>({});
const selectedPreset = ref<string>('');
const showApiKey = ref(false);
const loading = ref(false);
const testing = ref(false);
const saving = ref(false);
const testResult = ref<{ success: boolean; message: string } | null>(null);
const saveResult = ref<{ success: boolean; message: string } | null>(null);

const presetOptions = [
  { label: 'DashScope (通义千问)', value: 'DashScope (通义千问)' },
  { label: 'OpenAI', value: 'OpenAI' },
  { label: 'DeepSeek', value: 'DeepSeek' },
  { label: 'Ollama (本地)', value: 'Ollama (本地)' },
  { label: 'Moonshot (月之暗面)', value: 'Moonshot (月之暗面)' },
  { label: '自定义', value: '' },
];

async function loadConfig() {
  loading.value = true;
  try {
    const res = await fetch(`${API_BASE}/api/v1/model/config`);
    const data = await res.json();
    if (data.success) {
      presets.value = data.presets || {};
      // 反填配置（api_key 是脱敏的，不回填到输入框）
      config.base_url = data.config.base_url;
      config.model_name = data.config.model_name;
      config.temperature = data.config.temperature;
      // api_key 显示脱敏值，但输入框留空让用户填写新 key
      config.api_key = '';
    }
  } catch (e) {
    console.error('加载配置失败:', e);
  } finally {
    loading.value = false;
  }
}

function onPresetChange() {
  if (selectedPreset.value && presets.value[selectedPreset.value]) {
    const p = presets.value[selectedPreset.value];
    config.base_url = p.base_url;
    config.model_name = p.model_name;
  }
}

async function testConnection() {
  if (!config.api_key || !config.base_url || !config.model_name) {
    testResult.value = { success: false, message: '请填写完整配置' };
    return;
  }
  testing.value = true;
  testResult.value = null;
  try {
    const res = await fetch(`${API_BASE}/api/v1/model/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: config.api_key,
        base_url: config.base_url,
        model_name: config.model_name,
      }),
    });
    testResult.value = await res.json();
  } catch (e: any) {
    testResult.value = { success: false, message: `请求失败: ${e.message}` };
  } finally {
    testing.value = false;
  }
}

async function saveConfig() {
  saving.value = true;
  saveResult.value = null;
  try {
    const body: any = {};
    if (config.api_key) body.api_key = config.api_key;
    body.base_url = config.base_url;
    body.model_name = config.model_name;
    body.temperature = config.temperature;
    if (selectedPreset.value) body.preset = selectedPreset.value;

    const res = await fetch(`${API_BASE}/api/v1/model/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    saveResult.value = await res.json();
    if (saveResult.value?.success) {
      // 保存成功后清空 api_key 输入（安全）
      config.api_key = '';
    }
  } catch (e: any) {
    saveResult.value = { success: false, message: `保存失败: ${e.message}` };
  } finally {
    saving.value = false;
  }
}

async function resetConfig() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/model/reset`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      config.base_url = data.config.base_url;
      config.model_name = data.config.model_name;
      config.temperature = data.config.temperature;
      config.api_key = '';
      saveResult.value = { success: true, message: '已恢复默认配置' };
    }
  } catch (e: any) {
    saveResult.value = { success: false, message: `重置失败: ${e.message}` };
  }
}

onMounted(loadConfig);
</script>

<template>
  <div class="settings-panel">
    <div class="settings-header">
      <h2>⚙️ 模型设置</h2>
      <p class="subtitle">配置 LLM 模型提供商和 API Key</p>
    </div>

    <div class="settings-body">
      <!-- 预设提供商 -->
      <div class="form-group">
        <label>预设提供商</label>
        <select v-model="selectedPreset" @change="onPresetChange" class="form-select">
          <option v-for="opt in presetOptions" :key="opt.value" :value="opt.value">
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- Base URL -->
      <div class="form-group">
        <label>API Base URL</label>
        <input
          v-model="config.base_url"
          type="text"
          class="form-input"
          placeholder="https://api.openai.com/v1"
        />
        <span class="hint">OpenAI 兼容的接口地址</span>
      </div>

      <!-- Model Name -->
      <div class="form-group">
        <label>模型名称</label>
        <input
          v-model="config.model_name"
          type="text"
          class="form-input"
          placeholder="gpt-4o"
        />
        <span class="hint">如 qwen-max, gpt-4o, deepseek-chat</span>
      </div>

      <!-- API Key -->
      <div class="form-group">
        <label>API Key</label>
        <div class="api-key-input">
          <input
            v-model="config.api_key"
            :type="showApiKey ? 'text' : 'password'"
            class="form-input"
            placeholder="输入新的 API Key（留空则不更新）"
          />
          <button class="toggle-vis" @click="showApiKey = !showApiKey" type="button">
            {{ showApiKey ? '🙈' : '👁️' }}
          </button>
        </div>
      </div>

      <!-- Temperature -->
      <div class="form-group">
        <label>温度 ({{ config.temperature.toFixed(2) }})</label>
        <input
          v-model.number="config.temperature"
          type="range"
          min="0"
          max="1"
          step="0.05"
          class="form-range"
        />
        <div class="range-labels">
          <span>精确 (0)</span>
          <span>随机 (1)</span>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div class="actions">
        <button
          class="btn btn-test"
          @click="testConnection"
          :disabled="testing || !config.api_key"
        >
          {{ testing ? '测试中...' : '🔗 测试连接' }}
        </button>
        <button class="btn btn-save" @click="saveConfig" :disabled="saving">
          {{ saving ? '保存中...' : '💾 保存配置' }}
        </button>
        <button class="btn btn-reset" @click="resetConfig">
          🔄 恢复默认
        </button>
      </div>

      <!-- 测试结果 -->
      <div v-if="testResult" :class="['result', testResult.success ? 'success' : 'error']">
        {{ testResult.success ? '✅' : '❌' }} {{ testResult.message }}
      </div>

      <!-- 保存结果 -->
      <div v-if="saveResult" :class="['result', saveResult.success ? 'success' : 'error']">
        {{ saveResult.success ? '✅' : '❌' }} {{ saveResult.message }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-panel {
  max-width: 680px;
  margin: 0 auto;
  padding: 2rem;
}

.settings-header {
  margin-bottom: 2rem;
}

.settings-header h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: #1e293b;
}

.subtitle {
  color: #64748b;
  font-size: 0.875rem;
  margin: 0;
}

.settings-body {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-group label {
  font-weight: 600;
  font-size: 0.875rem;
  color: #334155;
}

.form-input, .form-select {
  padding: 0.75rem 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.625rem;
  font-size: 0.9375rem;
  background: white;
  color: #1e293b;
  transition: border-color 0.2s;
}

.form-input:focus, .form-select:focus {
  outline: none;
  border-color: #8b5cf6;
  box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
}

.hint {
  font-size: 0.75rem;
  color: #94a3b8;
}

.api-key-input {
  display: flex;
  gap: 0.5rem;
}

.api-key-input .form-input {
  flex: 1;
}

.toggle-vis {
  padding: 0 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.625rem;
  background: #f8fafc;
  cursor: pointer;
  font-size: 1.125rem;
}

.form-range {
  width: 100%;
  accent-color: #8b5cf6;
}

.range-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #94a3b8;
}

.actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 0.625rem;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-test {
  background: #f0fdf4;
  color: #16a34a;
  border: 1px solid #bbf7d0;
}

.btn-test:hover:not(:disabled) {
  background: #dcfce7;
}

.btn-save {
  background: linear-gradient(135deg, #3b82f6, #06b6d4);
  color: white;
  box-shadow: 0 4px 12px -3px rgba(59, 130, 246, 0.4);
}

.btn-save:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px -4px rgba(59, 130, 246, 0.5);
}

.btn-reset {
  background: #f8fafc;
  color: #64748b;
  border: 1px solid #e2e8f0;
}

.btn-reset:hover {
  background: #f1f5f9;
}

.result {
  padding: 0.875rem 1rem;
  border-radius: 0.625rem;
  font-size: 0.875rem;
  font-weight: 500;
}

.result.success {
  background: #f0fdf4;
  color: #16a34a;
  border: 1px solid #bbf7d0;
}

.result.error {
  background: #fef2f2;
  color: #dc2626;
  border: 1px solid #fecaca;
}
</style>
