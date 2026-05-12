<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  turnover?: number;
  high?: number;
  low?: number;
  open?: number;
  prevClose?: number;
  dayHigh?: number;
  dayLow?: number;
  yearHigh?: number;
  yearLow?: number;
  marketCap?: number;
}

const emit = defineEmits<{
  analyze: [symbol: string, name: string];
}>();

const API_BASE = 'http://localhost:8000';
const CACHE_TTL = 30000; // 30秒缓存

interface CacheEntry {
  data: Stock[];
  timestamp: number;
}

const activeTab = ref<'zh' | 'us'>('zh');
const stocks = ref<Stock[]>([]);
const loading = ref(false);
const error = ref('');
const lastUpdated = ref('');
const cache = ref<Record<string, CacheEntry>>({});

async function fetchMarketData(force = false) {
  const now = Date.now();
  const cached = cache.value[activeTab.value];

  // 有有效缓存且非强制刷新时直接复用
  if (!force && cached && now - cached.timestamp < CACHE_TTL) {
    stocks.value = cached.data;
    lastUpdated.value = new Date(cached.timestamp).toLocaleTimeString('zh-CN');
    return;
  }

  loading.value = true;
  error.value = '';
  try {
    const res = await fetch(`${API_BASE}/api/v1/market?market=${activeTab.value}&limit=50`);
    const data = await res.json();
    if (data.success) {
      stocks.value = data.stocks || [];
      cache.value[activeTab.value] = { data: stocks.value, timestamp: Date.now() };
      lastUpdated.value = new Date().toLocaleTimeString('zh-CN');
    } else {
      error.value = data.error || '获取数据失败';
    }
  } catch (e: any) {
    error.value = '网络错误: ' + e.message;
  } finally {
    loading.value = false;
  }
}

function formatNumber(num: number): string {
  if (num >= 1e8) {
    return (num / 1e8).toFixed(2) + '亿';
  }
  if (num >= 1e4) {
    return (num / 1e4).toFixed(2) + '万';
  }
  return num.toLocaleString('zh-CN');
}

function formatPrice(price: number): string {
  return price.toFixed(2);
}

function formatChangePercent(pct: number): string {
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(2)}%`;
}

function getChangeClass(pct: number): string {
  if (pct > 0) return 'up';
  if (pct < 0) return 'down';
  return 'flat';
}

function handleAnalyze(stock: Stock) {
  emit('analyze', stock.symbol, stock.name);
}

watch(activeTab, () => {
  fetchMarketData();
});

onMounted(() => {
  fetchMarketData();
});
</script>

<template>
  <div class="market-panel">
    <!-- 标题栏 -->
    <div class="panel-header">
      <div class="header-left">
        <h2 class="panel-title">📊 实时行情</h2>
        <span v-if="lastUpdated" class="update-time">更新于 {{ lastUpdated }}</span>
      </div>
      <button class="refresh-btn" :disabled="loading" @click="fetchMarketData(true)">
        <span class="refresh-icon" :class="{ spinning: loading }">🔄</span>
        <span>{{ loading ? '加载中...' : '刷新' }}</span>
      </button>
    </div>

    <!-- Tab 切换 -->
    <div class="tab-bar">
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'zh' }"
        @click="activeTab = 'zh'"
      >
        🇨🇳 A股
      </button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'us' }"
        @click="activeTab = 'us'"
      >
        🇺🇸 美股
      </button>
    </div>

    <!-- 错误提示 -->
    <div v-if="error" class="error-box">
      <span class="error-icon">⚠️</span>
      <span>{{ error }}</span>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading && stocks.length === 0" class="loading-box">
      <div class="spinner"></div>
      <span>正在加载行情数据...</span>
    </div>

    <!-- 数据表格 -->
    <div v-else class="table-wrapper">
      <table class="stock-table">
        <thead>
          <tr>
            <th class="col-name">名称</th>
            <th class="col-symbol">代码</th>
            <th class="col-price">现价</th>
            <th class="col-change">涨跌额</th>
            <th class="col-pct">涨跌幅</th>
            <th class="col-volume">成交量</th>
            <th v-if="activeTab === 'zh'" class="col-turnover">成交额</th>
            <th v-if="activeTab === 'us'" class="col-cap">市值</th>
            <th class="col-action">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="stock in stocks"
            :key="stock.symbol"
            class="stock-row"
          >
            <td class="col-name">
              <div class="stock-name">{{ stock.name }}</div>
            </td>
            <td class="col-symbol">
              <span class="symbol-tag">{{ stock.symbol }}</span>
            </td>
            <td class="col-price">
              <span class="price">{{ formatPrice(stock.price) }}</span>
            </td>
            <td class="col-change">
              <span :class="getChangeClass(stock.change)">
                {{ stock.change >= 0 ? '+' : '' }}{{ formatPrice(stock.change) }}
              </span>
            </td>
            <td class="col-pct">
              <span class="pct-badge" :class="getChangeClass(stock.changePercent)">
                {{ formatChangePercent(stock.changePercent) }}
              </span>
            </td>
            <td class="col-volume">{{ formatNumber(stock.volume) }}</td>
            <td v-if="activeTab === 'zh'" class="col-turnover">
              {{ stock.turnover ? formatNumber(stock.turnover) : '-' }}
            </td>
            <td v-if="activeTab === 'us'" class="col-cap">
              {{ stock.marketCap ? formatNumber(stock.marketCap) : '-' }}
            </td>
            <td class="col-action">
              <button class="analyze-btn" @click="handleAnalyze(stock)">
                🤖 分析
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.market-panel {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  max-width: 72rem;
  margin: 0 auto;
  width: 100%;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.25rem;
}

.header-left {
  display: flex;
  align-items: baseline;
  gap: 1rem;
}

.panel-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1e293b;
  margin: 0;
}

.update-time {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.4);
}

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 0.625rem;
  color: #2563eb;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.refresh-btn:hover:not(:disabled) {
  background: rgba(59, 130, 246, 0.15);
  transform: translateY(-1px);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.refresh-icon {
  display: inline-block;
  transition: transform 0.3s;
}

.refresh-icon.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Tab 栏 */
.tab-bar {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
  padding: 0.375rem;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 0.75rem;
  width: fit-content;
}

.tab-btn {
  padding: 0.625rem 1.25rem;
  background: transparent;
  border: none;
  border-radius: 0.625rem;
  font-size: 0.9375rem;
  font-weight: 500;
  color: rgba(0, 0, 0, 0.55);
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn:hover {
  color: #1e293b;
}

.tab-btn.active {
  background: white;
  color: #7c3aed;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  font-weight: 600;
}

/* 错误提示 */
.error-box {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: 0.75rem;
  color: #dc2626;
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

/* 加载状态 */
.loading-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 4rem;
  color: rgba(0, 0, 0, 0.45);
}

.spinner {
  width: 2rem;
  height: 2rem;
  border: 2px solid rgba(0, 0, 0, 0.08);
  border-top-color: #7c3aed;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* 表格 */
.table-wrapper {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 1rem;
  overflow: hidden;
}

.stock-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.stock-table thead {
  background: rgba(0, 0, 0, 0.02);
}

.stock-table th {
  padding: 0.875rem 1rem;
  text-align: left;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.5);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  white-space: nowrap;
}

.stock-table td {
  padding: 0.875rem 1rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
  white-space: nowrap;
}

.stock-row:hover {
  background: rgba(139, 92, 246, 0.03);
}

.stock-row:last-child td {
  border-bottom: none;
}

.col-name {
  min-width: 8rem;
}

.col-symbol {
  min-width: 5rem;
}

.col-price {
  min-width: 5rem;
}

.col-change {
  min-width: 5rem;
}

.col-pct {
  min-width: 5rem;
}

.col-volume {
  min-width: 6rem;
}

.col-turnover {
  min-width: 6rem;
}

.col-cap {
  min-width: 6rem;
}

.col-action {
  min-width: 5rem;
  text-align: center;
}

.stock-name {
  font-weight: 600;
  color: #1e293b;
}

.symbol-tag {
  font-family: monospace;
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.5);
  background: rgba(0, 0, 0, 0.04);
  padding: 0.125rem 0.5rem;
  border-radius: 0.375rem;
}

.price {
  font-weight: 600;
  color: #1e293b;
  font-variant-numeric: tabular-nums;
}

/* 涨跌颜色 - 国内习惯：红涨绿跌 */
.up {
  color: #dc2626;
}

.down {
  color: #16a34a;
}

.flat {
  color: #6b7280;
}

.pct-badge {
  display: inline-block;
  padding: 0.25rem 0.625rem;
  border-radius: 0.375rem;
  font-size: 0.8125rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.pct-badge.up {
  background: rgba(220, 38, 38, 0.1);
}

.pct-badge.down {
  background: rgba(22, 163, 74, 0.1);
}

.pct-badge.flat {
  background: rgba(107, 114, 128, 0.1);
}

.analyze-btn {
  padding: 0.375rem 0.75rem;
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(124, 58, 237, 0.1));
  border: 1px solid rgba(139, 92, 246, 0.25);
  border-radius: 0.5rem;
  color: #7c3aed;
  font-size: 0.8125rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.analyze-btn:hover {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(124, 58, 237, 0.2));
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(139, 92, 246, 0.15);
}

/* 滚动条 */
.market-panel::-webkit-scrollbar {
  width: 6px;
}

.market-panel::-webkit-scrollbar-track {
  background: transparent;
}

.market-panel::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.1);
  border-radius: 3px;
}

.market-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.2);
}

/* 响应式 */
@media (max-width: 768px) {
  .market-panel {
    padding: 1rem;
  }

  .stock-table {
    font-size: 0.8125rem;
  }

  .stock-table th,
  .stock-table td {
    padding: 0.625rem 0.5rem;
  }

  .col-volume,
  .col-turnover,
  .col-cap {
    display: none;
  }
}
</style>
