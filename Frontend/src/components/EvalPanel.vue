<script setup lang="ts">
import { friendlyError } from '../utils/errors';
/**
 * 评测中心：只读展示 Eval/runs/ 的可靠性报告、单次运行时间线、报告原文与结构化日志。
 * 数据来自 /api/v1/eval/runs*。
 */
import { ref, computed, onMounted } from 'vue';
import type {
  EvalRunSummary,
  EvalReport,
  EvalTraceRun,
  EvalCaseSummary,
  EvalTraceEvent,
  EvalLogEntry,
  EvalLogLevel,
  EvalLogResponse,
} from '../types';
import MarkdownRender from './MarkdownRender.vue';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const runs = ref<EvalRunSummary[]>([]);
const loadingRuns = ref(false);
const selectedRunId = ref<string | null>(null);
const report = ref<EvalReport | null>(null);
const traces = ref<EvalTraceRun[]>([]);
const selectedTraceId = ref<string | null>(null);
const loadingDetail = ref(false);
const error = ref('');
const reportMd = ref('');
const reportAvailable = ref(true);
const reportOpen = ref(false);
const logRows = ref<EvalLogEntry[]>([]);
const logAvailable = ref(true);
const logSkipped = ref(0);
const logLevel = ref<'ALL' | EvalLogLevel>('ALL');
const expandedLog = ref<number | null>(null);

onMounted(loadRuns);

async function loadRuns() {
  loadingRuns.value = true;
  error.value = '';
  try {
    const res = await fetch(`${API_BASE}/api/v1/eval/runs`);
    const data = await res.json();
    runs.value = data.runs || [];
    const first = runs.value[0];
    if (first) await selectRun(first.run_id);
  } catch (e: any) {
    error.value = `加载评测列表失败：${friendlyError(e)}`;
  } finally {
    loadingRuns.value = false;
  }
}

async function selectRun(runId: string) {
  selectedRunId.value = runId;
  selectedTraceId.value = null;
  loadingDetail.value = true;
  error.value = '';
  reportOpen.value = false;
  expandedLog.value = null;
  try {
    const [r1, r2, r3, r4] = await Promise.all([
      fetch(`${API_BASE}/api/v1/eval/runs/${runId}`).then((r) => r.json()),
      fetch(`${API_BASE}/api/v1/eval/runs/${runId}/trace`).then((r) => r.json()),
      fetch(`${API_BASE}/api/v1/eval/runs/${runId}/report-md`).then((r) => r.json()),
      fetch(`${API_BASE}/api/v1/eval/runs/${runId}/log`).then((r) => r.json() as Promise<EvalLogResponse>),
    ]);
    report.value = r1.report || null;
    traces.value = r2.runs || [];
    selectedTraceId.value = traces.value[0]?.run_id ?? null;
    reportMd.value = r3.markdown || '';
    reportAvailable.value = r3.available !== false;
    logRows.value = r4.rows || [];
    logAvailable.value = r4.available !== false;
    logSkipped.value = r4.skipped_lines || 0;
  } catch (e: any) {
    error.value = `加载评测详情失败：${friendlyError(e)}`;
  } finally {
    loadingDetail.value = false;
  }
}

const selectedRun = computed(() => runs.value.find((r) => r.run_id === selectedRunId.value) || null);

const selectedTrace = computed(
  () => traces.value.find((t) => t.run_id === selectedTraceId.value) || null
);

const runsByCase = computed(() => {
  const map: Record<string, EvalTraceRun[]> = {};
  for (const t of traces.value) {
    const arr = map[t.case_id];
    if (arr) arr.push(t);
    else map[t.case_id] = [t];
  }
  return map;
});

function pct(v: number | null | undefined) {
  return v == null ? '—' : `${Math.round(v * 100)}%`;
}
function secs(v: number | null | undefined) {
  return v == null ? '—' : `${v.toFixed(1)}s`;
}
function ciText(c: EvalCaseSummary, kind: 'pipeline' | 'strict') {
  const arr = kind === 'pipeline' ? c.pipeline_ci95 : c.strict_ci95;
  if (!arr) return '';
  return `${Math.round(arr[0] * 100)}–${Math.round(arr[1] * 100)}%`;
}
function fmtCount(v: Record<string, number> | undefined) {
  const e = Object.entries(v || {});
  return e.length ? e.map(([k, n]) => `${k} × ${n}`).join('，') : '无';
}
function runIdx(t: EvalTraceRun) {
  return t.run_idx ?? '?';
}

const filteredLog = computed(() =>
  logLevel.value === 'ALL' ? logRows.value : logRows.value.filter((r) => r.level === logLevel.value)
);
const logCounts = computed(() => {
  const c: Record<string, number> = { INFO: 0, WARN: 0, ERROR: 0 };
  for (const r of logRows.value) c[r.level] = (c[r.level] || 0) + 1;
  return c;
});
function agentBrief(agents: EvalLogEntry['agents']) {
  if (!agents) return '—';
  return Object.entries(agents)
    .map(
      ([n, a]) =>
        `${n}：${a.vote ?? '未知'}${a.error ? '（异常兜底）' : ''}${a.parse_fallback ? '（解析兜底）' : ''}`
    )
    .join('；');
}

// 时间线：agent 用 start/end 区间，其它阶段用时间点
const timeline = computed(() => {
  const run = selectedTrace.value;
  if (!run) return { total: 1, spans: [] as any[], points: [] as EvalTraceEvent[] };
  const evs = run.events || [];
  const total = Math.max(run.latency_s ?? 0, ...evs.map((e) => e.t), 0.1);
  const doneByAgent = new Map<string, { t: number; vote?: string }>();
  for (const e of evs) {
    if (e.stage === 'agent_done' && e.agent) doneByAgent.set(e.agent, { t: e.t, vote: e.vote });
  }
  const seen = new Set<string>();
  const spans: any[] = [];
  const points: EvalTraceEvent[] = [];
  for (const e of evs) {
    if (e.stage === 'agent_start' && e.agent && !seen.has(e.agent)) {
      seen.add(e.agent);
      const done = doneByAgent.get(e.agent);
      spans.push({ label: e.agent, start: e.t, end: done?.t ?? e.t, vote: done?.vote });
    } else if (['preprocess', 'route', 'chat', 'clarify', 'voting'].includes(e.stage)) {
      points.push(e);
    }
  }
  return { total, spans, points };
});

// 按同一次运行中出现的顺序分配颜色，保证同批次内不同 Agent 颜色互异
const AGENT_COLORS = ['#f5a524', '#2563eb', '#0891b2', '#059669', '#d97706', '#db2777'];
function agentColor(index: number) {
  return AGENT_COLORS[index % AGENT_COLORS.length] ?? '#f5a524';
}
function barStyle(start: number, end: number, color: string) {
  const total = timeline.value.total || 1;
  return {
    left: `${Math.min(100, (start / total) * 100)}%`,
    width: `${Math.max(0.8, ((end - start) / total) * 100)}%`,
    background: color,
  };
}
function markerStyle(t: number) {
  const total = timeline.value.total || 1;
  return { left: `${Math.min(100, (t / total) * 100)}%` };
}
const STAGE_LABEL: Record<string, string> = {
  preprocess: '意图解析',
  route: '路由决策',
  chat: '闲聊回复',
  clarify: '澄清',
  voting: '投票汇总',
  agent_start: 'Agent 启动',
  agent_done: 'Agent 完成',
};
function stageLabel(e: EvalTraceEvent) {
  let s = STAGE_LABEL[e.stage] || e.stage;
  if (e.agent) s += ` · ${e.agent}`;
  if (e.reason) s += `：${e.reason}`;
  return s;
}
function verdict(strict: boolean | null, pipeline: boolean | null) {
  if (strict) return { text: '跑得对', cls: 'ok' };
  if (pipeline) return { text: '能跑通', cls: 'warn' };
  return { text: '失败', cls: 'bad' };
}
</script>

<template>
  <div class="eval">
    <header class="eval-head">
      <div>
        <h1>评测中心</h1>
        <p class="lede">
          K 次重复运行的可靠性统计。<strong>能跑通</strong>指无异常、状态合法且有结果；
          <strong>跑得对</strong>在此基础上要求路由符合预期、无降级 Agent、投票值合法。
          结果只衡量可靠性，不代表决策准确率。
        </p>
      </div>
    </header>

    <div v-if="error" class="alert">{{ error }}</div>

    <div class="layout">
      <!-- 左：批次列表 -->
      <aside class="side">
        <div class="side-head">
          <span>评测批次</span>
          <button class="btn-ghost" :disabled="loadingRuns" @click="loadRuns">
            {{ loadingRuns ? '加载中' : '刷新' }}
          </button>
        </div>

        <p v-if="!loadingRuns && runs.length === 0" class="muted small">
          暂无评测数据。运行 <code>uv run python -m Eval.reliability --k 5</code> 生成。
        </p>

        <button
          v-for="r in runs"
          :key="r.run_id"
          class="run-card"
          :class="{ active: r.run_id === selectedRunId }"
          @click="selectRun(r.run_id)"
        >
          <div class="run-card-id">{{ r.run_id }}</div>
          <div class="muted small">K = {{ r.k }} · {{ r.total_runs }} 次运行 · {{ r.case_count }} 个用例</div>
          <div class="run-card-stats">
            <span>能跑通 <b>{{ r.pipeline_ok }}/{{ r.total_runs }}</b></span>
            <span>跑得对 <b :class="r.strict_ok === r.total_runs ? 'ok-text' : 'warn-text'">{{ r.strict_ok }}/{{ r.total_runs }}</b></span>
          </div>
        </button>
      </aside>

      <!-- 右：详情 -->
      <section class="main">
        <div v-if="loadingDetail" class="muted">加载中…</div>

        <template v-else-if="report">
          <div class="meta-line">
            <span>{{ selectedRun?.run_id }}</span>
            <span>生成于 {{ report.generated_at.replace('T', ' ') }}</span>
            <span>K = {{ report.k }}，共 {{ report.total_runs }} 次</span>
          </div>

          <!-- 用例指标 -->
          <div class="case-grid">
            <article v-for="c in report.cases" :key="c.case_id" class="card">
              <header class="card-head">
                <h3>{{ c.case_id }}</h3>
                <span class="muted small">{{ c.runs }} 次运行</span>
              </header>

              <div class="stats">
                <div class="stat">
                  <div class="stat-label">能跑通</div>
                  <div class="stat-value">{{ c.pipeline_ok }}<span class="stat-of">/{{ c.runs }}</span></div>
                  <div class="stat-sub">{{ pct(c.pipeline_rate) }} · 95% CI {{ ciText(c, 'pipeline') }}</div>
                </div>
                <div class="stat">
                  <div class="stat-label">跑得对</div>
                  <div class="stat-value">{{ c.strict_ok }}<span class="stat-of">/{{ c.runs }}</span></div>
                  <div class="stat-sub">{{ pct(c.strict_rate) }} · 95% CI {{ ciText(c, 'strict') }}</div>
                </div>
                <div class="stat">
                  <div class="stat-label">延迟（p50）</div>
                  <div class="stat-value">{{ secs(c.latency_p50_s) }}</div>
                  <div class="stat-sub">p95 {{ secs(c.latency_p95_s) }} · 均值 {{ secs(c.latency_mean_s) }}</div>
                </div>
                <div class="stat">
                  <div class="stat-label">降级率</div>
                  <div class="stat-value">{{ pct(c.degraded_run_rate) }}</div>
                  <div class="stat-sub">路由：{{ fmtCount(c.routes) }}</div>
                </div>
              </div>

              <div v-if="Object.keys(c.final_votes || {}).length" class="votes">
                最终投票：{{ fmtCount(c.final_votes) }}
              </div>
            </article>
          </div>

          <!-- 时间线 -->
          <section class="card">
            <header class="card-head">
              <h3>单次运行时间线</h3>
              <span class="muted small">Agent 区间按开始与结束时间绘制</span>
            </header>

            <div class="picker">
              <div v-for="(group, cid) in runsByCase" :key="cid" class="picker-group">
                <span class="picker-label">{{ cid }}</span>
                <button
                  v-for="t in group"
                  :key="t.run_id"
                  class="chip"
                  :class="{ active: t.run_id === selectedTraceId, fail: t.pipeline_ok === false }"
                  @click="selectedTraceId = t.run_id"
                >
                  第 {{ runIdx(t) }} 次 · {{ secs(t.latency_s) }}
                </button>
              </div>
            </div>

            <template v-if="selectedTrace">
              <div class="trace-meta">
                <span class="mono">{{ selectedTrace.run_id }}</span>
                <span>路由 <b>{{ selectedTrace.route }}</b></span>
                <span>耗时 <b>{{ secs(selectedTrace.latency_s) }}</b></span>
                <span v-if="selectedTrace.final_vote">投票 <b>{{ selectedTrace.final_vote }}</b></span>
                <span class="tag" :class="verdict(selectedTrace.strict_ok, selectedTrace.pipeline_ok).cls">
                  {{ verdict(selectedTrace.strict_ok, selectedTrace.pipeline_ok).text }}
                </span>
              </div>
              <p class="query">问题：{{ selectedTrace.query }}</p>

              <div class="timeline">
                <div v-for="(s, i) in timeline.spans" :key="'s' + i" class="tl-row">
                  <div class="tl-name">
                    {{ s.label }}<span v-if="s.vote" class="tl-vote"> · {{ s.vote }}</span>
                  </div>
                  <div class="tl-track">
                    <div class="tl-bar" :style="barStyle(s.start, s.end, agentColor(i))"></div>
                  </div>
                  <span class="tl-time">{{ s.start.toFixed(1) }}–{{ s.end.toFixed(1) }}s</span>
                </div>
                <div v-for="(p, i) in timeline.points" :key="'p' + i" class="tl-row">
                  <div class="tl-name muted">{{ stageLabel(p) }}</div>
                  <div class="tl-track">
                    <div class="tl-marker" :style="markerStyle(p.t)"></div>
                  </div>
                  <span class="tl-time">{{ p.t.toFixed(2) }}s</span>
                </div>
              </div>
              <div class="axis">
                <span></span>
                <div class="axis-track">
                  <span>0s</span>
                  <span>{{ timeline.total.toFixed(1) }}s</span>
                </div>
                <span></span>
              </div>

              <ul v-if="selectedTrace.reasons.length" class="reasons">
                <li v-for="(r, i) in selectedTrace.reasons" :key="i">{{ r }}</li>
              </ul>
            </template>
          </section>

          <!-- 报告原文 -->
          <section class="card">
            <header class="card-head">
              <h3>报告原文</h3>
              <button class="btn-ghost" @click="reportOpen = !reportOpen">
                {{ reportOpen ? '收起' : '展开' }}
              </button>
            </header>
            <div v-if="reportOpen" class="report-body">
              <MarkdownRender v-if="reportMd" :content="reportMd" />
              <p v-else class="muted small">该批次没有 report.md。</p>
            </div>
            <p v-else class="muted small">report.md 的完整内容，展开查看。</p>
          </section>

          <!-- 结构化日志 -->
          <section class="card">
            <header class="card-head">
              <h3>运行日志</h3>
              <span class="muted small">
                run_log.jsonl · 共 {{ logRows.length }} 条<template v-if="logSkipped">，跳过 {{ logSkipped }} 行损坏数据</template>
              </span>
            </header>

            <div v-if="logAvailable" class="segmented">
              <button :class="{ on: logLevel === 'ALL' }" @click="logLevel = 'ALL'">全部 {{ logRows.length }}</button>
              <button :class="{ on: logLevel === 'INFO' }" @click="logLevel = 'INFO'">INFO {{ logCounts.INFO }}</button>
              <button :class="{ on: logLevel === 'WARN' }" @click="logLevel = 'WARN'">WARN {{ logCounts.WARN }}</button>
              <button :class="{ on: logLevel === 'ERROR' }" @click="logLevel = 'ERROR'">ERROR {{ logCounts.ERROR }}</button>
            </div>

            <p v-if="!logAvailable" class="muted small">
              该批次生成于结构化日志功能之前，没有 run_log.jsonl。
            </p>
            <p v-else-if="filteredLog.length === 0" class="muted small">没有符合条件的日志。</p>
            <table v-else class="log">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>级别</th>
                  <th>事件</th>
                  <th>用例</th>
                  <th>次数</th>
                  <th>路由</th>
                  <th>投票</th>
                  <th class="num">耗时</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="(r, i) in filteredLog" :key="i">
                  <tr class="log-row" @click="expandedLog = expandedLog === i ? null : i">
                    <td class="mono">{{ r.ts.slice(11, 23) }}</td>
                    <td><span class="level" :class="r.level.toLowerCase()">{{ r.level }}</span></td>
                    <td class="mono">{{ r.event }}</td>
                    <td>{{ r.case_id ?? '' }}</td>
                    <td>{{ r.run_idx ?? '' }}</td>
                    <td>{{ r.route ?? '' }}</td>
                    <td>{{ r.final_vote ?? '' }}</td>
                    <td class="num">{{ r.latency_s !== undefined ? secs(r.latency_s) : '' }}</td>
                  </tr>
                  <tr v-if="expandedLog === i" class="log-detail">
                    <td colspan="8">
                      <p v-if="r.agents">Agent：{{ agentBrief(r.agents) }}</p>
                      <p v-if="r.reasons && r.reasons.length">原因：{{ r.reasons.join('；') }}</p>
                      <pre>{{ JSON.stringify(r, null, 2) }}</pre>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
          </section>
        </template>

        <div v-else class="muted">请选择左侧的评测批次。</div>
      </section>
    </div>
  </div>
</template>

<style scoped>
/* 与应用主色保持一致：紫色强调、slate 文本、细边框、无 emoji */
.eval {
  --ink: #f4f4f5;
  --ink-2: #d4d4d8;
  --muted: #8b8f98;
  --line: #27272a;
  --accent: #f5a524;
  --ok: #34d399;
  --warn: #fbbf24;
  --bad: #ff6b6b;
  flex: 1;
  overflow-y: auto;
  padding: 2rem 2.25rem 3rem;
  color: var(--ink);
  font-size: 0.9rem;
  line-height: 1.55;
}

h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
}

.lede {
  margin: 0.4rem 0 0;
  max-width: 62rem;
  color: var(--muted);
  font-size: 0.85rem;
}

.lede strong {
  color: var(--ink-2);
  font-weight: 600;
}

.muted {
  color: var(--muted);
}

.small {
  font-size: 0.78rem;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem;
}

.ok-text {
  color: var(--ok);
}

.warn-text {
  color: var(--warn);
}

code {
  background: #18181b;
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  font-size: 0.78rem;
}

.alert {
  margin-top: 1rem;
  padding: 0.7rem 1rem;
  border: 1px solid rgba(239,68,68,0.35);
  background: rgba(239,68,68,0.10);
  color: var(--bad);
  border-radius: 8px;
  font-size: 0.85rem;
}

/* 布局 */
.layout {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-top: 1.5rem;
}

/* 批次横向排列在顶部，主区域占满宽度 */
.side {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 0.75rem;
}

.side-head {
  grid-column: 1 / -1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--muted);
  padding: 0 0.1rem;
}

.btn-ghost {
  border: 1px solid var(--line);
  background: #0b0b0c;
  color: var(--ink-2);
  border-radius: 6px;
  padding: 0.25rem 0.7rem;
  font-size: 0.78rem;
  cursor: pointer;
}

.btn-ghost:hover {
  border-color: #3f3f46;
}

.btn-ghost:disabled {
  opacity: 0.6;
  cursor: default;
}

.run-card {
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  padding: 0.75rem 0.9rem;
  background: #0b0b0c;
  border: 1px solid var(--line);
  border-radius: 10px;
  cursor: pointer;
  font: inherit;
  color: inherit;
}

.run-card:hover {
  border-color: #3f3f46;
}

.run-card.active {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent) inset;
}

.run-card-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8rem;
  font-weight: 600;
}

.run-card-stats {
  display: flex;
  gap: 0.9rem;
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.15rem;
}

.run-card-stats b {
  color: var(--ink-2);
  font-weight: 600;
}

.main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.meta-line {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.2rem;
  color: var(--muted);
  font-size: 0.8rem;
}

.meta-line span:first-child {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--ink-2);
}

/* 卡片 */
.card {
  background: #0b0b0c;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1.1rem 1.25rem;
}

.card-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 0.9rem;
}

.case-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 1rem;
}

.stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.9rem 1.25rem;
}

.stat-label {
  font-size: 0.75rem;
  color: var(--muted);
}

.stat-value {
  font-size: 1.4rem;
  font-weight: 650;
  color: var(--ink);
  line-height: 1.2;
  margin-top: 0.1rem;
  white-space: nowrap;
}

.stat-of {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--muted);
}

.stat-sub {
  font-size: 0.72rem;
  color: var(--muted);
  margin-top: 0.15rem;
  white-space: nowrap;
}

.votes {
  margin-top: 0.9rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--line);
  font-size: 0.78rem;
  color: var(--ink-2);
}

/* 时间线 */
.picker {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.picker-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
}

.picker-label {
  width: 7.5rem;
  font-size: 0.78rem;
  color: var(--muted);
}

.chip {
  border: 1px solid var(--line);
  background: #0b0b0c;
  border-radius: 999px;
  padding: 0.2rem 0.7rem;
  font-size: 0.75rem;
  color: var(--ink-2);
  cursor: pointer;
}

.chip.active {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(245,165,36,0.08);
  font-weight: 600;
}

.chip.fail {
  border-color: rgba(239,68,68,0.35);
  color: var(--bad);
}

.trace-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 1.1rem;
  font-size: 0.8rem;
  color: var(--muted);
}

.trace-meta b {
  color: var(--ink-2);
  font-weight: 600;
}

.query {
  margin: 0.5rem 0 1rem;
  font-size: 0.82rem;
  color: var(--ink-2);
}

.tag {
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
}

.tag.ok {
  background: rgba(34,197,94,0.12);
  color: var(--ok);
}

.tag.warn {
  background: rgba(245,158,11,0.10);
  color: var(--warn);
}

.tag.bad {
  background: rgba(239,68,68,0.10);
  color: var(--bad);
}

.timeline {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.tl-row {
  display: grid;
  grid-template-columns: 200px minmax(0, 1fr) 84px;
  align-items: center;
  gap: 0.75rem;
}

.tl-name {
  font-size: 0.8rem;
  color: var(--ink-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tl-vote {
  color: var(--muted);
}

.tl-track {
  position: relative;
  height: 22px;
  background: #121214;
  border: 1px solid var(--line);
  border-radius: 6px;
}

.tl-bar {
  position: absolute;
  top: 3px;
  bottom: 3px;
  border-radius: 4px;
  min-width: 3px;
  opacity: 0.9;
}

.tl-marker {
  position: absolute;
  top: 4px;
  bottom: 4px;
  width: 2px;
  background: #475569;
  transform: translateX(-1px);
}

.tl-time {
  font-size: 0.72rem;
  color: var(--ink-2);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  text-align: right;
  white-space: nowrap;
}

.axis {
  display: grid;
  grid-template-columns: 200px minmax(0, 1fr) 84px;
  gap: 0.75rem;
  margin-top: 0.4rem;
  font-size: 0.7rem;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.axis-track {
  display: flex;
  justify-content: space-between;
}

.reasons {
  margin: 0.9rem 0 0;
  padding-left: 1.1rem;
  font-size: 0.78rem;
  color: var(--warn);
}

/* 报告原文 */
.report-body {
  max-height: 520px;
  overflow: auto;
  padding: 0.25rem 0.1rem 0;
  border-top: 1px solid var(--line);
  padding-top: 0.9rem;
}

/* 分段控件 */
.segmented {
  display: inline-flex;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 0.8rem;
}

.segmented button {
  border: none;
  background: #0b0b0c;
  padding: 0.35rem 0.85rem;
  font-size: 0.78rem;
  color: var(--muted);
  cursor: pointer;
  border-right: 1px solid var(--line);
}

.segmented button:last-child {
  border-right: none;
}

.segmented button.on {
  background: #18181b;
  color: var(--ink);
  font-weight: 600;
}

/* 日志表 */
.log {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}

.log th,
.log td {
  padding: 0.5rem 0.6rem;
  text-align: left;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}

.log th {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--muted);
  background: #121214;
}

.log .num {
  text-align: right;
  white-space: nowrap;
}

.log-row {
  cursor: pointer;
}

.log-row:hover td {
  background: #121214;
}

.level {
  display: inline-block;
  min-width: 3.4rem;
  text-align: center;
  padding: 0.05rem 0.4rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 600;
  background: #18181b;
  color: var(--muted);
}

.level.warn {
  background: rgba(245,158,11,0.10);
  color: var(--warn);
}

.level.error {
  background: rgba(239,68,68,0.10);
  color: var(--bad);
}

.log-detail td {
  background: #121214;
  font-size: 0.76rem;
  color: var(--ink-2);
}

.log-detail p {
  margin: 0.2rem 0;
}

.log-detail pre {
  margin: 0.5rem 0 0;
  padding: 0.6rem 0.75rem;
  background: #0b0b0c;
  border: 1px solid var(--line);
  border-radius: 6px;
  font-size: 0.72rem;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--ink-2);
}

@media (max-width: 960px) {
  .tl-row,
  .axis {
    grid-template-columns: 130px minmax(0, 1fr) 76px;
  }
}
</style>
