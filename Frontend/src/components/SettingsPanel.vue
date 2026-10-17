<script setup lang="ts">
import { API_BASE, apiFetch, getApiToken, setApiToken } from '../utils/api';
import { friendlyError } from '../utils/errors';
import { ref, reactive, onMounted } from 'vue';



// ── Tab 切换 ──────────────────────────────────────────────
const activeTab = ref<'model' | 'agents' | 'access'>('model');

// ── 访问令牌（保存在本机浏览器；后端配置 FINGENT_API_TOKEN 时需要）──
const tokenInput = ref('');
const tokenSaved = ref(!!getApiToken());
const tokenStatus = ref<{ ok: boolean; text: string } | null>(null);
const tokenTesting = ref(false);

function saveToken() {
  const value = tokenInput.value.trim();
  if (!value) {
    tokenStatus.value = { ok: false, text: '请先输入令牌' };
    return;
  }
  setApiToken(value);
  tokenInput.value = '';
  tokenSaved.value = true;
  tokenStatus.value = { ok: true, text: '已保存到本机浏览器' };
}

function clearToken() {
  setApiToken('');
  tokenInput.value = '';
  tokenSaved.value = false;
  tokenStatus.value = { ok: true, text: '已清除本机令牌' };
}

async function testToken() {
  tokenTesting.value = true;
  tokenStatus.value = null;
  try {
    // 受保护接口：能返回 200 说明令牌（或本机免令牌）有效
    const res = await apiFetch(`${API_BASE}/api/v1/agents`);
    if (res.ok) {
      tokenStatus.value = { ok: true, text: '鉴权通过，可以访问受保护接口' };
    } else {
      const data = await res.json().catch(() => ({}));
      tokenStatus.value = { ok: false, text: data.error || `HTTP ${res.status}` };
    }
  } catch (e) {
    tokenStatus.value = { ok: false, text: friendlyError(e) };
  } finally {
    tokenTesting.value = false;
  }
}

// ── 模型设置 ──────────────────────────────────────────────
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
    const res = await apiFetch(`${API_BASE}/api/v1/model/config`);
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
  if (selectedPreset.value) {
    const p = presets.value[selectedPreset.value];
    if (!p) return;
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
    const res = await apiFetch(`${API_BASE}/api/v1/model/test`, {
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
    testResult.value = { success: false, message: `请求失败: ${friendlyError(e)}` };
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

    const res = await apiFetch(`${API_BASE}/api/v1/model/config`, {
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
    saveResult.value = { success: false, message: `保存失败: ${friendlyError(e)}` };
  } finally {
    saving.value = false;
  }
}

async function resetConfig() {
  try {
    const res = await apiFetch(`${API_BASE}/api/v1/model/reset`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      config.base_url = data.config.base_url;
      config.model_name = data.config.model_name;
      config.temperature = data.config.temperature;
      config.api_key = '';
      saveResult.value = { success: true, message: '已恢复默认配置' };
    }
  } catch (e: any) {
    saveResult.value = { success: false, message: `重置失败: ${friendlyError(e)}` };
  }
}

// ── Agent 角色管理 ────────────────────────────────────────
interface AgentMeta {
  name: string;
  description: string;
  allowed_tools: string[];
  builtin: boolean;
  editable: boolean;
}

const agents = ref<AgentMeta[]>([]);
const selectedAgent = ref<string>('');
const skillContent = ref<string>('');
const skillOriginal = ref<string>('');
const skillLoading = ref(false);
const skillSaving = ref(false);
const skillMessage = ref<{ success: boolean; message: string } | null>(null);
const reloading = ref(false);

const isDirty = () => skillContent.value !== skillOriginal.value;

async function loadAgents() {
  try {
    const res = await apiFetch(`${API_BASE}/api/v1/agents`);
    const data = await res.json();
    if (data.success) {
      agents.value = data.agents || [];
      const first = agents.value[0];
      if (!selectedAgent.value && first) {
        await selectAgent(first.name);
      }
    }
  } catch (e: any) {
    skillMessage.value = { success: false, message: `加载 agent 列表失败: ${friendlyError(e)}` };
  }
}

async function selectAgent(name: string) {
  if (isDirty() && !confirm('当前编辑未保存，切换会丢失修改。确定切换吗？')) {
    return;
  }
  selectedAgent.value = name;
  skillLoading.value = true;
  skillMessage.value = null;
  try {
    const res = await apiFetch(`${API_BASE}/api/v1/agents/${name}`);
    const data = await res.json();
    if (data.success) {
      skillContent.value = data.content || '';
      skillOriginal.value = data.content || '';
    } else {
      skillMessage.value = { success: false, message: data.error || '读取失败' };
    }
  } catch (e: any) {
    skillMessage.value = { success: false, message: `读取失败: ${friendlyError(e)}` };
  } finally {
    skillLoading.value = false;
  }
}

async function saveSkill() {
  if (!selectedAgent.value) return;
  skillSaving.value = true;
  skillMessage.value = null;
  try {
    const res = await apiFetch(`${API_BASE}/api/v1/agents/${selectedAgent.value}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: skillContent.value }),
    });
    const data = await res.json();
    skillMessage.value = {
      success: !!data.success,
      message: data.message || data.error || (data.success ? '已保存' : '保存失败'),
    };
    if (data.success) {
      skillOriginal.value = skillContent.value;
      await loadAgents(); // 刷新描述等元数据
    }
  } catch (e: any) {
    skillMessage.value = { success: false, message: `保存失败: ${friendlyError(e)}` };
  } finally {
    skillSaving.value = false;
  }
}

async function reloadAgents() {
  reloading.value = true;
  skillMessage.value = null;
  try {
    const res = await apiFetch(`${API_BASE}/api/v1/agents/reload`, { method: 'POST' });
    const data = await res.json();
    skillMessage.value = {
      success: !!data.success,
      message: data.message || data.error || '',
    };
    if (data.success) await loadAgents();
  } catch (e: any) {
    skillMessage.value = { success: false, message: `重载失败: ${friendlyError(e)}` };
  } finally {
    reloading.value = false;
  }
}

onMounted(() => {
  loadConfig();
  loadAgents();
});
</script>

<template>
  <div class="settings-panel">
    <div class="settings-header">
      <h2>系统设置</h2>
      <p class="subtitle">配置 LLM 模型，或管理各 Agent 的角色人设（SKILL.md）</p>
    </div>

    <!-- Tab 切换 -->
    <div class="tabs">
      <button
        :class="['tab', { active: activeTab === 'model' }]"
        @click="activeTab = 'model'"
      >
        模型设置
      </button>
      <button
        :class="['tab', { active: activeTab === 'agents' }]"
        @click="activeTab = 'agents'"
      >
        Agent 角色管理
      </button>
      <button
        :class="['tab', { active: activeTab === 'access' }]"
        @click="activeTab = 'access'"
      >
        访问令牌
      </button>
    </div>

    <!-- ═══════════ Tab 3: 访问令牌 ═══════════ -->
    <div v-show="activeTab === 'access'" class="settings-body">
      <div class="form-group">
        <label>API 访问令牌</label>
        <input
          v-model="tokenInput"
          type="password"
          class="form-input"
          autocomplete="off"
          :placeholder="tokenSaved ? '已保存（输入新值可覆盖）' : '粘贴后端 FINGENT_API_TOKEN 的值'"
        />
        <span class="hint">
          后端配置了 FINGENT_API_TOKEN 时，写入、对话、模型配置等接口需要它；
          未配置且后端只在本机运行时可留空。令牌只保存在本机浏览器。
        </span>
      </div>

      <div class="actions">
        <button class="btn btn-test" @click="testToken" :disabled="tokenTesting">
          {{ tokenTesting ? '验证中...' : '验证令牌' }}
        </button>
        <button class="btn btn-save" @click="saveToken" :disabled="!tokenInput.trim()">
          保存令牌
        </button>
        <button class="btn btn-reset" @click="clearToken" :disabled="!tokenSaved">
          清除令牌
        </button>
      </div>

      <div v-if="tokenStatus" :class="['result', tokenStatus.ok ? 'success' : 'error']">
        {{ tokenStatus.ok ? '✓' : '✕' }} {{ tokenStatus.text }}
      </div>
    </div>

    <!-- ═══════════ Tab 1: 模型设置 ═══════════ -->
    <div v-show="activeTab === 'model'" class="settings-body">
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
            {{ showApiKey ? '隐藏' : '显示' }}
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
          {{ testing ? '测试中...' : '测试连接' }}
        </button>
        <button class="btn btn-save" @click="saveConfig" :disabled="saving">
          {{ saving ? '保存中...' : '保存配置' }}
        </button>
        <button class="btn btn-reset" @click="resetConfig">
          恢复默认
        </button>
      </div>

      <!-- 测试结果 -->
      <div v-if="testResult" :class="['result', testResult.success ? 'success' : 'error']">
        {{ testResult.success ? '✓' : '✕' }} {{ testResult.message }}
      </div>

      <!-- 保存结果 -->
      <div v-if="saveResult" :class="['result', saveResult.success ? 'success' : 'error']">
        {{ saveResult.success ? '✓' : '✕' }} {{ saveResult.message }}
      </div>
    </div>

    <!-- ═══════════ Tab 2: Agent 角色管理 ═══════════ -->
    <div v-show="activeTab === 'agents'" class="settings-body">
      <div class="agent-toolbar">
        <select
          v-model="selectedAgent"
          @change="selectAgent(selectedAgent)"
          class="form-select agent-select"
        >
          <option v-for="a in agents" :key="a.name" :value="a.name">
            {{ a.builtin ? '内置' : '自定义' }} {{ a.name }}
          </option>
        </select>
        <button class="btn btn-reset" @click="reloadAgents" :disabled="reloading">
          {{ reloading ? '重载中...' : '重载技能目录' }}
        </button>
        <button
          class="btn btn-save"
          @click="saveSkill"
          :disabled="skillSaving || skillLoading || !isDirty() || !selectedAgent"
        >
          {{ skillSaving ? '保存中...' : '保存并生效' }}
        </button>
      </div>

      <div v-if="selectedAgent" class="agent-meta">
        <div class="meta-line">
          <span class="meta-label">描述</span>
          {{ agents.find(a => a.name === selectedAgent)?.description }}
        </div>
        <div class="meta-line">
          <span class="meta-label">工具</span>
          <code class="tool-chip">{{ (agents.find(a => a.name === selectedAgent)?.allowed_tools || []).join(' ') || '（全部工具）' }}</code>
        </div>
        <div class="meta-line hint">
          身份、分析协议与输出 JSON 格式都在下方 SKILL.md 中；保存后自动备份原文件并热重建，新对话立即生效。
        </div>
      </div>

      <div class="editor-wrap">
        <textarea
          v-model="skillContent"
          class="skill-editor"
          spellcheck="false"
          :disabled="skillLoading"
          placeholder="加载 SKILL.md ..."
        ></textarea>
        <div class="editor-statusbar">
          <span>{{ skillLoading ? '加载中...' : `${skillContent.length} 字符` }}</span>
          <span v-if="isDirty()" class="dirty-dot">● 未保存</span>
        </div>
      </div>

      <div v-if="skillMessage" :class="['result', skillMessage.success ? 'success' : 'error']">
        {{ skillMessage.success ? '✓' : '✕' }} {{ skillMessage.message }}
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
  margin-bottom: 1.25rem;
}

.settings-header h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: #e4e4e7;
}

.subtitle {
  color: #8b8f98;
  font-size: 0.875rem;
  margin: 0;
}

/* ── Tabs ── */
.tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid #27272a;
}

.tab {
  padding: 0.625rem 1.25rem;
  border: none;
  background: transparent;
  font-size: 0.875rem;
  font-weight: 600;
  color: #8b8f98;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: all 0.2s;
}

.tab:hover {
  color: #d4d4d8;
}

.tab.active {
  color: #f5a524;
  border-bottom-color: #f5a524;
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
  color: #d4d4d8;
}

.form-input, .form-select {
  padding: 0.75rem 1rem;
  border: 1px solid #27272a;
  border-radius: 0.625rem;
  font-size: 0.9375rem;
  background: #0b0b0c;
  color: #e4e4e7;
  transition: border-color 0.2s;
}

.form-input:focus, .form-select:focus {
  outline: none;
  border-color: #f5a524;
  box-shadow: 0 0 0 3px rgba(245,165,36,0.1);
}

.hint {
  font-size: 0.75rem;
  color: #71717a;
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
  border: 1px solid #27272a;
  border-radius: 0.625rem;
  background: #121214;
  cursor: pointer;
  font-size: 1.125rem;
}

.form-range {
  width: 100%;
  accent-color: #f5a524;
}

.range-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #71717a;
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
  background: rgba(34,197,94,0.10);
  color: #22c55e;
  border: 1px solid rgba(34,197,94,0.35);
}

.btn-test:hover:not(:disabled) {
  background: rgba(34,197,94,0.16);
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
  background: #121214;
  color: #8b8f98;
  border: 1px solid #27272a;
}

.btn-reset:hover {
  background: #18181b;
}

.result {
  padding: 0.875rem 1rem;
  border-radius: 0.625rem;
  font-size: 0.875rem;
  font-weight: 500;
}

.result.success {
  background: rgba(34,197,94,0.10);
  color: #22c55e;
  border: 1px solid rgba(34,197,94,0.35);
}

.result.error {
  background: rgba(239,68,68,0.10);
  color: #ff4d4f;
  border: 1px solid rgba(239,68,68,0.35);
}

/* ── Agent 管理 ── */
.agent-toolbar {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
}

.agent-toolbar .btn {
  padding: 0.625rem 1rem;
  font-size: 0.8125rem;
}

.agent-select {
  flex: 1;
  min-width: 200px;
}

.agent-meta {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
  padding: 0.875rem 1rem;
  background: #121214;
  border: 1px solid #27272a;
  border-radius: 0.625rem;
  font-size: 0.8125rem;
  color: #b4b4bb;
}

.meta-line {
  line-height: 1.5;
}

.meta-label {
  display: inline-block;
  min-width: 2.5rem;
  font-weight: 600;
  color: #d4d4d8;
}

.tool-chip {
  font-size: 0.75rem;
  background: #121214;
  color: #4f46e5;
  padding: 0.125rem 0.5rem;
  border-radius: 0.375rem;
  word-break: break-all;
}

.editor-wrap {
  border: 1px solid #27272a;
  border-radius: 0.625rem;
  overflow: hidden;
  background: #0b0b0c;
}

.skill-editor {
  width: 100%;
  min-height: 380px;
  max-height: 55vh;
  padding: 1rem;
  border: none;
  resize: vertical;
  font-family: 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.8125rem;
  line-height: 1.6;
  color: #e4e4e7;
  background: #0b0b0c;
  box-sizing: border-box;
}

.skill-editor:focus {
  outline: none;
}

.editor-statusbar {
  display: flex;
  justify-content: space-between;
  padding: 0.375rem 1rem;
  font-size: 0.75rem;
  color: #71717a;
  background: #121214;
  border-top: 1px solid #27272a;
}

.dirty-dot {
  color: #fbbf24;
  font-weight: 600;
}
</style>
