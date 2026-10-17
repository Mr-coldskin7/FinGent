<script setup lang="ts">
import { API_BASE, apiFetch } from '../utils/api';
/**
 * 规则管理：列出当前用户的纠正规则，可启用/禁用/删除。
 * 规则来自分析结果下的「纠正规则」反馈；启用的规则会注入对应 Agent 的提示词，禁用后不再注入。
 * 数据来自 /api/v1/memory/rules*。
 */
import { ref, computed, watch, onMounted } from 'vue';
import { friendlyError } from '../utils/errors';
import type { MemoryRule } from '../types';
import { agentLabel as agentShortLabel } from '../utils/agents';

const props = defineProps<{ userId: string }>();


// 与后端 WEIGHT_RULE_PROMOTE_THRESHOLD 保持一致：触发次数达到该值视为强规则
const STRONG_THRESHOLD = 3;

const rules = ref<MemoryRule[]>([]);
const loading = ref(false);
const error = ref('');
const filter = ref<'all' | 'active' | 'inactive'>('all');
const busyId = ref<number | null>(null);
const confirmId = ref<number | null>(null);
const notice = ref('');

const SOURCE_LABEL: Record<string, string> = {
  explicit_feedback: '用户纠正',
  self_mined: '自动挖掘',
};

const activeCount = computed(() => rules.value.filter((r) => r.active).length);
const strongCount = computed(() => rules.value.filter((r) => r.active && r.trigger_count >= STRONG_THRESHOLD).length);
const visibleRules = computed(() => {
  if (filter.value === 'active') return rules.value.filter((r) => r.active);
  if (filter.value === 'inactive') return rules.value.filter((r) => !r.active);
  return rules.value;
});

function agentLabel(name: string): string {
  return name === 'ALL' ? '全部 Agent' : agentShortLabel(name);
}

function formatTime(ts: string): string {
  return ts ? ts.slice(0, 16).replace('T', ' ') : '-';
}

async function load() {
  loading.value = true;
  error.value = '';
  try {
    const res = await apiFetch(
      `${API_BASE}/api/v1/memory/rules?user_id=${encodeURIComponent(props.userId)}&include_inactive=true`,
    );
    const data = await res.json();
    if (!data.success) throw new Error(data.error || '加载失败');
    rules.value = data.rules || [];
  } catch (e) {
    error.value = `加载规则失败：${friendlyError(e)}`;
  } finally {
    loading.value = false;
  }
}

async function toggle(rule: MemoryRule) {
  busyId.value = rule.id;
  notice.value = '';
  try {
    const res = await apiFetch(`${API_BASE}/api/v1/memory/rules/${rule.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: props.userId, active: !rule.active }),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || '操作失败');
    rule.active = !rule.active;
    notice.value = rule.active ? '已启用，下次分析起注入对应 Agent。' : '已禁用，下次分析起不再注入。';
  } catch (e) {
    notice.value = `操作失败：${friendlyError(e)}`;
  } finally {
    busyId.value = null;
  }
}

async function remove(rule: MemoryRule) {
  busyId.value = rule.id;
  notice.value = '';
  try {
    const res = await apiFetch(
      `${API_BASE}/api/v1/memory/rules/${rule.id}?user_id=${encodeURIComponent(props.userId)}`,
      { method: 'DELETE' },
    );
    const data = await res.json();
    if (!data.success) throw new Error(data.error || '删除失败');
    rules.value = rules.value.filter((r) => r.id !== rule.id);
    notice.value = '规则已删除。';
  } catch (e) {
    notice.value = `删除失败：${friendlyError(e)}`;
  } finally {
    busyId.value = null;
    confirmId.value = null;
  }
}

watch(() => props.userId, load);
onMounted(load);
</script>

<template>
  <div class="rules">
    <header class="rules-head">
      <div>
        <h1>规则管理</h1>
        <p class="lede">
          分析结果下点击「纠正规则」会生成规则。<strong>启用</strong>的规则会注入对应 Agent 的提示词，
          <strong>禁用</strong>后保留记录但不再生效，<strong>删除</strong>不可恢复。
        </p>
      </div>
      <button class="ghost" :disabled="loading" @click="load">{{ loading ? '加载中' : '刷新' }}</button>
    </header>

    <section class="stats">
      <div class="stat">
        <span class="stat-label">规则总数</span>
        <span class="stat-value">{{ rules.length }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">已启用</span>
        <span class="stat-value ok">{{ activeCount }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">强规则（触发 ≥ {{ STRONG_THRESHOLD }} 次）</span>
        <span class="stat-value accent">{{ strongCount }}</span>
      </div>
    </section>

    <div class="toolbar">
      <div class="segmented" role="tablist">
        <button :class="{ on: filter === 'all' }" @click="filter = 'all'">全部</button>
        <button :class="{ on: filter === 'active' }" @click="filter = 'active'">已启用</button>
        <button :class="{ on: filter === 'inactive' }" @click="filter = 'inactive'">已禁用</button>
      </div>
      <span v-if="notice" class="notice">{{ notice }}</span>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>

    <div v-if="!loading && !error && rules.length === 0" class="empty">
      <p>暂无规则。</p>
      <p class="muted">在「智能对话」的分析结果下点击「纠正规则」，填写纠正内容即可添加。</p>
    </div>

    <div v-else-if="visibleRules.length === 0" class="empty">
      <p class="muted">当前筛选下没有规则。</p>
    </div>

    <div v-else class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="col-rule">规则内容</th>
            <th>适用 Agent</th>
            <th class="num">触发</th>
            <th>来源</th>
            <th>创建时间</th>
            <th>状态</th>
            <th class="col-actions">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in visibleRules" :key="r.id" :class="{ off: !r.active }">
            <td class="col-rule">
              <span class="rule-text">{{ r.rule_text }}</span>
              <span v-if="r.active && r.trigger_count >= STRONG_THRESHOLD" class="tag strong">强规则</span>
            </td>
            <td>{{ agentLabel(r.agent_name) }}</td>
            <td class="num">{{ r.trigger_count }}</td>
            <td>{{ SOURCE_LABEL[r.source] || r.source }}</td>
            <td class="time">{{ formatTime(r.created_at) }}</td>
            <td>
              <button
                class="switch"
                :class="{ on: r.active }"
                :disabled="busyId === r.id"
                :aria-label="r.active ? '禁用规则' : '启用规则'"
                @click="toggle(r)"
              >
                <span class="knob" />
                <span class="switch-text">{{ r.active ? '启用' : '禁用' }}</span>
              </button>
            </td>
            <td class="col-actions">
              <template v-if="confirmId === r.id">
                <button class="danger" :disabled="busyId === r.id" @click="remove(r)">确认删除</button>
                <button class="link" @click="confirmId = null">取消</button>
              </template>
              <button v-else class="link danger-text" :disabled="busyId === r.id" @click="confirmId = r.id">
                删除
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.rules {
  --bg: #0b0b0c;
  --panel: #121214;
  --line: #27272a;
  --ink: #f4f4f5;
  --muted: #8b8f98;
  --accent: #f5a524;
  --ok: #4ade80;
  --bad: #f87171;

  max-width: 1080px;
  margin: 0 auto;
  padding: 28px 24px 48px;
  color: var(--ink);
}
.rules-head {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;
  margin-bottom: 20px;
}
h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; letter-spacing: 0.01em; }
.lede { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.6; max-width: 680px; }
.lede strong { color: var(--ink); font-weight: 500; }

.ghost, .segmented button, .link {
  font: inherit; cursor: pointer;
}
.ghost {
  background: transparent; color: var(--ink); border: 1px solid var(--line);
  border-radius: 8px; padding: 7px 14px; font-size: 13px; white-space: nowrap;
}
.ghost:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.ghost:disabled { opacity: 0.5; cursor: default; }

.stats {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px;
}
.stat {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; display: flex; flex-direction: column; gap: 6px;
}
.stat-label { font-size: 12px; color: var(--muted); }
.stat-value { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
.stat-value.ok { color: var(--ok); }
.stat-value.accent { color: var(--accent); }

.toolbar {
  display: flex; justify-content: space-between; align-items: center; gap: 12px;
  margin-bottom: 12px; min-height: 32px;
}
.segmented {
  display: inline-flex; background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 3px;
}
.segmented button {
  background: transparent; border: 0; color: var(--muted); padding: 5px 12px;
  border-radius: 6px; font-size: 13px;
}
.segmented button.on { background: #27272a; color: var(--ink); }
.notice { font-size: 13px; color: var(--muted); }

.error-box {
  border: 1px solid #7f1d1d; background: rgba(127, 29, 29, 0.15); color: #fca5a5;
  padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 12px;
}

.empty {
  border: 1px dashed var(--line); border-radius: 12px; padding: 40px 20px; text-align: center;
}
.empty p { margin: 4px 0; }
.muted { color: var(--muted); font-size: 13px; }

.table-wrap {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow-x: auto;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  text-align: left; font-weight: 500; color: var(--muted); font-size: 12px;
  padding: 11px 14px; border-bottom: 1px solid var(--line); white-space: nowrap;
}
td {
  padding: 12px 14px; border-bottom: 1px solid var(--line); vertical-align: middle;
}
tr:last-child td { border-bottom: 0; }
tr.off .rule-text, tr.off td { color: #6b6f78; }
.col-rule { min-width: 260px; }
.rule-text { line-height: 1.5; }
.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.time { color: var(--muted); white-space: nowrap; }
.col-actions { white-space: nowrap; text-align: right; }

.tag {
  display: inline-block; margin-left: 8px; font-size: 11px; padding: 1px 7px;
  border-radius: 999px; vertical-align: middle;
}
.tag.strong { color: var(--accent); border: 1px solid rgba(245, 165, 36, 0.45); }

.switch {
  display: inline-flex; align-items: center; gap: 8px; background: transparent; border: 0;
  color: var(--muted); font: inherit; font-size: 13px; cursor: pointer; padding: 0;
}
.switch .knob {
  width: 30px; height: 18px; border-radius: 999px; background: #3f3f46; position: relative;
  transition: background 0.15s;
}
.switch .knob::after {
  content: ''; position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
  border-radius: 50%; background: #e4e4e7; transition: transform 0.15s;
}
.switch.on .knob { background: var(--accent); }
.switch.on .knob::after { transform: translateX(12px); background: #0a0a0a; }
.switch.on .switch-text { color: var(--ink); }
.switch:disabled { opacity: 0.5; cursor: default; }

.link {
  background: transparent; border: 0; color: var(--muted); font-size: 13px; padding: 4px 6px;
}
.link:hover { color: var(--ink); }
.danger-text { color: #d4d4d8; }
.danger-text:hover { color: var(--bad); }
.danger {
  background: transparent; border: 1px solid #7f1d1d; color: var(--bad);
  border-radius: 6px; font: inherit; font-size: 12px; padding: 4px 10px; cursor: pointer; margin-right: 4px;
}
.danger:hover:not(:disabled) { background: rgba(248, 113, 113, 0.12); }
.danger:disabled { opacity: 0.5; cursor: default; }

@media (max-width: 720px) {
  .stats { grid-template-columns: 1fr; }
  .rules-head { flex-direction: column; }
}
</style>
