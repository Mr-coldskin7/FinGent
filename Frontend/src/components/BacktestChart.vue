<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue';

interface PriceData {
  date: string;
  price: number;
  portfolio_value: number;
}

interface TradeMarker {
  date: string;
  type: 'buy' | 'sell';
  price: number;
  size: number;
  value: number;
}

interface ChartStatistics {
  initial_value: number;
  final_value: number;
  return_pct: number;
  total_trades: number;
  buy_count: number;
  sell_count: number;
}

interface ChartData {
  symbol: string;
  price_data: PriceData[];
  trade_markers: TradeMarker[];
  statistics: ChartStatistics;
}

const props = defineProps<{
  data?: ChartData;
  loading?: boolean;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const tooltip = ref({
  show: false,
  x: 0,
  y: 0,
  content: ''
});

// 图表配置
const CHART_CONFIG = {
  padding: { top: 40, right: 60, bottom: 60, left: 60 },
  colors: {
    priceLine: '#3b82f6',
    priceFill: 'rgba(59, 130, 246, 0.1)',
    buyMarker: '#22c55e',
    sellMarker: '#ef4444',
    portfolioLine: '#8b5cf6',
    portfolioFill: 'rgba(139, 92, 246, 0.1)',
    grid: 'rgba(255, 255, 255, 0.1)',
    text: 'rgba(255, 255, 255, 0.6)'
  }
};

// 格式化数字
const formatNumber = (num: number, decimals = 2) => {
  return num.toFixed(decimals);
};

// 格式化日期
const formatDate = (dateStr: string) => {
  return dateStr.slice(5); // 显示 MM-DD
};

// 绘制图表
const drawChart = () => {
  if (!canvasRef.value || !props.data?.price_data?.length) return;
  
  const canvas = canvasRef.value;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // 设置Canvas尺寸 (HiDPI 支持)
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  
  const width = rect.width;
  const height = rect.height;
  const { padding, colors } = CHART_CONFIG;
  
  // 清空画布
  ctx.clearRect(0, 0, width, height);
  
  const { price_data, trade_markers, statistics } = props.data;
  
  // 计算绘图区域
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  // 数据范围
  const prices = price_data.map(d => d.price);
  const portfolioValues = price_data.map(d => d.portfolio_value);
  const minPrice = Math.min(...prices) * 0.98;
  const maxPrice = Math.max(...prices) * 1.02;
  const minPortfolio = Math.min(...portfolioValues) * 0.98;
  const maxPortfolio = Math.max(...portfolioValues) * 1.02;
  
  // 比例尺
  const xScale = chartWidth / (price_data.length - 1);
  const priceScale = chartHeight / (maxPrice - minPrice);
  const portfolioScale = chartHeight / (maxPortfolio - minPortfolio);
  
  // 辅助函数：坐标转换
  const getX = (index: number) => padding.left + index * xScale;
  const getYPrice = (price: number) => padding.top + chartHeight - (price - minPrice) * priceScale;
  const getYPortfolio = (value: number) => padding.top + chartHeight - (value - minPortfolio) * portfolioScale;
  
  // ========== 绘制网格 ==========
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  
  // 水平网格线
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  
  // 垂直网格线 (每周一条)
  for (let i = 0; i < price_data.length; i += 5) {
    const x = getX(i);
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();
  }
  
  ctx.setLineDash([]);
  
  // ========== 绘制价格曲线 ==========
  ctx.beginPath();
  ctx.strokeStyle = colors.priceLine;
  ctx.lineWidth = 2;
  
  price_data.forEach((d, i) => {
    const x = getX(i);
    const y = getYPrice(d.price);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  
  // 价格区域填充
  ctx.beginPath();
  ctx.fillStyle = colors.priceFill;
  price_data.forEach((d, i) => {
    const x = getX(i);
    const y = getYPrice(d.price);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.lineTo(getX(price_data.length - 1), height - padding.bottom);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.closePath();
  ctx.fill();
  
  // ========== 绘制资金曲线 ==========
  ctx.beginPath();
  ctx.strokeStyle = colors.portfolioLine;
  ctx.lineWidth = 2;
  
  price_data.forEach((d, i) => {
    const x = getX(i);
    const y = getYPortfolio(d.portfolio_value);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  
  // ========== 绘制买卖点 ==========
  const dateIndexMap = new Map(price_data.map((d, i) => [d.date, i]));
  
  trade_markers.forEach(trade => {
    const idx = dateIndexMap.get(trade.date);
    if (idx === undefined) return;
    
    const x = getX(idx);
    const y = getYPrice(trade.price);
    
    if (trade.type === 'buy') {
      // 买入：绿色三角形
      ctx.fillStyle = colors.buyMarker;
      ctx.beginPath();
      ctx.moveTo(x, y - 15);
      ctx.lineTo(x - 8, y - 3);
      ctx.lineTo(x + 8, y - 3);
      ctx.closePath();
      ctx.fill();
      
      // 白色边框
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      // 卖出：红色倒三角形
      ctx.fillStyle = colors.sellMarker;
      ctx.beginPath();
      ctx.moveTo(x, y + 15);
      ctx.lineTo(x - 8, y + 3);
      ctx.lineTo(x + 8, y + 3);
      ctx.closePath();
      ctx.fill();
      
      // 白色边框
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
  
  // ========== 绘制坐标轴 ==========
  ctx.strokeStyle = colors.text;
  ctx.lineWidth = 1;
  
  // X轴
  ctx.beginPath();
  ctx.moveTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  
  // Y轴 (左 - 价格)
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.stroke();
  
  // Y轴 (右 - 资金)
  ctx.beginPath();
  ctx.moveTo(width - padding.right, padding.top);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  
  // ========== 绘制标签 ==========
  ctx.fillStyle = colors.text;
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  
  // X轴日期标签
  for (let i = 0; i < price_data.length; i += 5) {
    const x = getX(i);
    ctx.fillText(formatDate(price_data[i].date), x, height - padding.bottom + 20);
  }
  
  // Y轴价格标签 (左侧)
  ctx.textAlign = 'right';
  ctx.fillStyle = colors.priceLine;
  for (let i = 0; i <= 5; i++) {
    const price = minPrice + (maxPrice - minPrice) * (1 - i / 5);
    const y = padding.top + (chartHeight / 5) * i;
    ctx.fillText('$' + formatNumber(price, 0), padding.left - 10, y + 4);
  }
  
  // Y轴资金标签 (右侧)
  ctx.textAlign = 'left';
  ctx.fillStyle = colors.portfolioLine;
  for (let i = 0; i <= 5; i++) {
    const value = minPortfolio + (maxPortfolio - minPortfolio) * (1 - i / 5);
    const y = padding.top + (chartHeight / 5) * i;
    ctx.fillText('$' + formatNumber(value, 0), width - padding.right + 10, y + 4);
  }
  
  // ========== 绘制图例 ==========
  const legendY = 20;
  let legendX = width / 2 - 150;
  
  // 价格线
  ctx.strokeStyle = colors.priceLine;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(legendX, legendY);
  ctx.lineTo(legendX + 20, legendY);
  ctx.stroke();
  ctx.fillStyle = colors.text;
  ctx.textAlign = 'left';
  ctx.fillText('Price', legendX + 25, legendY + 4);
  
  legendX += 80;
  
  // 资金线
  ctx.strokeStyle = colors.portfolioLine;
  ctx.beginPath();
  ctx.moveTo(legendX, legendY);
  ctx.lineTo(legendX + 20, legendY);
  ctx.stroke();
  ctx.fillText('Portfolio', legendX + 25, legendY + 4);
  
  legendX += 100;
  
  // 买入标记
  ctx.fillStyle = colors.buyMarker;
  ctx.beginPath();
  ctx.moveTo(legendX + 10, legendY - 5);
  ctx.lineTo(legendX + 5, legendY + 5);
  ctx.lineTo(legendX + 15, legendY + 5);
  ctx.closePath();
  ctx.fill();
  ctx.fillText('Buy', legendX + 20, legendY + 4);
  
  legendX += 60;
  
  // 卖出标记
  ctx.fillStyle = colors.sellMarker;
  ctx.beginPath();
  ctx.moveTo(legendX + 10, legendY + 5);
  ctx.lineTo(legendX + 5, legendY - 5);
  ctx.lineTo(legendX + 15, legendY - 5);
  ctx.closePath();
  ctx.fill();
  ctx.fillText('Sell', legendX + 20, legendY + 4);
};

// 鼠标移动事件 - 显示tooltip
const handleMouseMove = (e: MouseEvent) => {
  if (!canvasRef.value || !props.data?.price_data?.length) return;
  
  const canvas = canvasRef.value;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  const { padding } = CHART_CONFIG;
  const chartWidth = rect.width - padding.left - padding.right;
  
  // 计算数据索引
  const dataIndex = Math.round((x - padding.left) / (chartWidth / (props.data.price_data.length - 1)));
  
  if (dataIndex >= 0 && dataIndex < props.data.price_data.length) {
    const data = props.data.price_data[dataIndex];
    tooltip.value = {
      show: true,
      x: e.clientX + 10,
      y: e.clientY - 10,
      content: `
        <div class="tooltip-date">${data.date}</div>
        <div class="tooltip-row"><span>Price:</span><span>$${formatNumber(data.price)}</span></div>
        <div class="tooltip-row"><span>Portfolio:</span><span>$${formatNumber(data.portfolio_value)}</span></div>
      `
    };
  } else {
    tooltip.value.show = false;
  }
};

const handleMouseLeave = () => {
  tooltip.value.show = false;
};

// 监听数据变化
watch(() => props.data, drawChart, { deep: true });

onMounted(() => {
  drawChart();
  window.addEventListener('resize', drawChart);
});

// 导出图表为图片
const exportChart = () => {
  if (!canvasRef.value) return;
  const link = document.createElement('a');
  link.download = `backtest_${props.data?.symbol}_${new Date().toISOString().slice(0, 10)}.png`;
  link.href = canvasRef.value.toDataURL();
  link.click();
};

// 收益率颜色
const returnColor = computed(() => {
  const pct = props.data?.statistics?.return_pct || 0;
  return pct >= 0 ? '#22c55e' : '#ef4444';
});
</script>

<template>
  <div class="backtest-chart">
    <div class="chart-header">
      <h3 class="chart-title">
        📈 {{ data?.symbol || 'Backtest' }} Chart
      </h3>
      <div v-if="data?.statistics" class="chart-stats">
        <span class="stat-item">
          Return: 
          <span class="stat-value" :style="{ color: returnColor }">
            {{ data.statistics.return_pct > 0 ? '+' : '' }}{{ formatNumber(data.statistics.return_pct) }}%
          </span>
        </span>
        <span class="stat-item">
          Trades: {{ data.statistics.total_trades }}
          <span class="trade-detail">({{ data.statistics.buy_count }} buy / {{ data.statistics.sell_count }} sell)</span>
        </span>
        <span class="stat-item">
          Final: ${{ formatNumber(data.statistics.final_value, 0) }}
        </span>
        <button class="export-btn" @click="exportChart" title="导出图片">
          💾
        </button>
      </div>
    </div>
    
    <div class="chart-container">
      <canvas
        v-if="data?.price_data?.length"
        ref="canvasRef"
        class="chart-canvas"
        @mousemove="handleMouseMove"
        @mouseleave="handleMouseLeave"
      />
      <div v-else-if="loading" class="chart-loading">
        <div class="loading-spinner"></div>
        <span>加载图表数据...</span>
      </div>
      <div v-else class="chart-empty">
        <span>暂无图表数据</span>
      </div>
    </div>
    
    <!-- Tooltip -->
    <div
      v-if="tooltip.show"
      class="chart-tooltip"
      :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
      v-html="tooltip.content"
    />
  </div>
</template>

<style scoped>
.backtest-chart {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 1rem;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.chart-title {
  font-size: 1rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.95);
  margin: 0;
}

.chart-stats {
  display: flex;
  gap: 1.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.stat-item {
  font-size: 0.8125rem;
  color: rgba(255, 255, 255, 0.6);
}

.stat-value {
  font-weight: 600;
  margin-left: 0.25rem;
}

.trade-detail {
  font-size: 0.75rem;
  opacity: 0.7;
}

.export-btn {
  padding: 0.375rem 0.625rem;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s;
}

.export-btn:hover {
  background: rgba(255, 255, 255, 0.12);
}

.chart-container {
  width: 100%;
  height: 400px;
  position: relative;
}

.chart-canvas {
  width: 100%;
  height: 100%;
  cursor: crosshair;
}

.chart-loading,
.chart-empty {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  color: rgba(255, 255, 255, 0.4);
  font-size: 0.875rem;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(139, 92, 246, 0.2);
  border-top-color: #8b5cf6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.chart-tooltip {
  position: fixed;
  background: rgba(0, 0, 0, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 0.5rem;
  padding: 0.75rem;
  pointer-events: none;
  z-index: 1000;
  font-size: 0.8125rem;
  min-width: 150px;
}

.chart-tooltip :deep(.tooltip-date) {
  color: rgba(255, 255, 255, 0.9);
  font-weight: 600;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.chart-tooltip :deep(.tooltip-row) {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.25rem;
  color: rgba(255, 255, 255, 0.7);
}

@media (max-width: 768px) {
  .chart-container {
    height: 300px;
  }
  
  .chart-stats {
    gap: 0.75rem;
  }
}
</style>
