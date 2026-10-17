<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue';
import { agentFullName } from '../utils/agents';

interface AgentProgressItem {
  name: string;
  status: 'running' | 'done' | 'error';
  vote?: string;
}

interface Progress {
  stage: string;
  agents: AgentProgressItem[];
  startedAt: number;
}

const props = defineProps<{ progress: Progress | null }>();

// ── 已耗时（秒）──────────────────────────────────────────
const elapsed = ref(0);
let timer: ReturnType<typeof setInterval> | null = null;

watch(
  () => props.progress?.startedAt,
  (startedAt) => {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (startedAt) {
      elapsed.value = Math.floor((Date.now() - startedAt) / 1000);
      timer = setInterval(() => {
        elapsed.value = Math.floor((Date.now() - startedAt) / 1000);
      }, 1000);
    }
  },
  { immediate: true }
);

onUnmounted(() => {
  if (timer) clearInterval(timer);
});

const elapsedText = computed(() => {
  const s = elapsed.value;
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`;
});

// ── 阶段文案与可见性 ──────────────────────────────────────
const STAGE_TEXT: Record<string, string> = {
  start: '正在连接分析引擎…',
  preprocess: '正在解析查询意图…',
  agents: '分析师团队并行工作中…',
  voting: '正在汇总各分析师投票…',
  chat: '正在组织回复…',
  clarify: '需要补充信息…',
  sector: '正在整理板块背景…',
};

const stageText = computed(() => STAGE_TEXT[props.progress?.stage || 'start'] || 'AI 正在分析...');

const agents = computed(() => props.progress?.agents || []);
const doneCount = computed(() => agents.value.filter(a => a.status !== 'running').length);
const multiAgent = computed(() => agents.value.length > 1);
const showAgents = computed(() => agents.value.length > 0);

const prettyName = agentFullName;
</script>

<template>
  <div class="loading-message">
    <div class="ai-avatar">
      <span class="avatar-icon"></span>
      <div class="avatar-pulse"></div>
    </div>
    <div class="content">
      <div class="loading-bubble">
        <div class="thinking-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <span class="loading-text">{{ stageText }}</span>
        <span class="elapsed">{{ elapsedText }}</span>
      </div>

      <!-- 多 Agent 进度清单：实时显示每位分析师的完成状态 -->
      <div v-if="showAgents" class="agent-progress">
        <div class="agent-progress-header">
          <span v-if="multiAgent">多 Agent 投票模式 · {{ doneCount }}/{{ agents.length }} 完成</span>
          <span v-else>单 Agent 分析模式</span>
        </div>
        <div class="agent-list">
          <div
            v-for="a in agents"
            :key="a.name"
            :class="['agent-item', a.status]"
          >
            <span class="agent-status-icon">
              <template v-if="a.status === 'running'">…</template>
              <template v-else-if="a.status === 'error'">!</template>
              <template v-else>✓</template>
            </span>
            <span class="agent-name">{{ prettyName(a.name) }}</span>
            <span v-if="a.status === 'done' && a.vote" class="agent-vote">{{ a.vote }}</span>
            <span v-else-if="a.status === 'running'" class="agent-running-text">分析中…</span>
            <span v-else-if="a.status === 'error'" class="agent-error-text">已降级</span>
          </div>
        </div>
      </div>

      <p v-else class="loading-subtitle">正在查询数据并生成报告</p>
    </div>
  </div>
</template>

<style scoped>
.loading-message {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  animation: slide-up 0.4s ease-out forwards;
}

.ai-avatar {
  position: relative;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  background: linear-gradient(135deg, #10b981, #14b8a6);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.avatar-icon {
  font-size: 1.125rem;
  position: relative;
  z-index: 1;
}

.avatar-pulse {
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  background: rgba(16, 185, 129, 0.3);
  animation: pulse-ring 2s ease-out infinite;
}

@keyframes pulse-ring {
  0% {
    transform: scale(1);
    opacity: 0.6;
  }
  100% {
    transform: scale(1.3);
    opacity: 0;
  }
}

.content {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 0;
}

.loading-bubble {
  display: flex;
  align-items: center;
  gap: 0.875rem;
  padding: 1rem 1.25rem;
  background: rgba(11,11,12,0.92);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 1.125rem;
  border-top-left-radius: 0.25rem;
  width: fit-content;
}

.thinking-dots {
  display: flex;
  gap: 0.25rem;
}

.thinking-dots span {
  width: 6px;
  height: 6px;
  background: linear-gradient(135deg, #3b82f6, #06b6d4);
  border-radius: 50%;
  animation: thinking 1.4s infinite ease-in-out;
}

.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes thinking {
  0%, 60%, 100% { 
    transform: translateY(0); 
    opacity: 0.4;
  }
  30% { 
    transform: translateY(-5px); 
    opacity: 1;
  }
}

.loading-text {
  font-size: 0.9375rem;
  color: rgba(255,255,255,0.65);
  font-weight: 500;
}

.elapsed {
  font-size: 0.75rem;
  color: rgba(255,255,255,0.4);
  font-variant-numeric: tabular-nums;
  padding-left: 0.5rem;
  border-left: 1px solid rgba(255,255,255,0.08);
}

/* ── Agent 进度清单 ── */
.agent-progress {
  background: rgba(11,11,12,0.92);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 0.875rem;
  padding: 0.75rem 1rem;
  animation: slide-up 0.3s ease-out forwards;
}

.agent-progress-header {
  font-size: 0.75rem;
  font-weight: 600;
  color: rgba(255,255,255,0.45);
  margin-bottom: 0.5rem;
}

.agent-list {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.agent-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8125rem;
  color: rgba(255,255,255,0.65);
}

.agent-item.done {
  color: rgba(255,255,255,0.55);
}

.agent-status-icon {
  width: 1.25rem;
  text-align: center;
  flex-shrink: 0;
}

.agent-item.running .agent-status-icon {
  animation: gentle-pulse 1.6s ease-in-out infinite;
}

@keyframes gentle-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.agent-name {
  font-weight: 500;
}

.agent-vote {
  margin-left: auto;
  font-size: 0.75rem;
  font-weight: 700;
  color: #34d399;
  background: rgba(16, 185, 129, 0.12);
  padding: 0.0625rem 0.5rem;
  border-radius: 0.375rem;
}

.agent-running-text,
.agent-error-text {
  margin-left: auto;
  font-size: 0.75rem;
  color: rgba(255,255,255,0.35);
}

.agent-error-text {
  color: #fbbf24;
}

.loading-subtitle {
  font-size: 0.75rem;
  color: rgba(255,255,255,0.4);
  margin: 0 0 0 0.5rem;
}

@keyframes slide-up {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
