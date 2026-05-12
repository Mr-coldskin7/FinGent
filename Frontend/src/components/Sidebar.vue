<script setup lang="ts">
defineProps<{
  currentView: 'chat' | 'backtest' | 'market';
}>();

const emit = defineEmits<{
  'new-chat': [];
  'quick-input': [text: string];
  'switch-view': [view: 'chat' | 'backtest' | 'market'];
}>();

const quickActions = [
  { icon: '📊', text: '贵州茅台', prompt: '分析一下贵州茅台' },
  { icon: '🚀', text: '英伟达', prompt: 'NVDA 最近怎么样' },
  { icon: '⚖️', text: '对比分析', prompt: '茅台和五粮液哪个好' },
  { icon: '💡', text: '投资建议', prompt: '推荐几只股票' },
];

const navItems = [
  { id: 'chat', icon: '💬', text: '智能对话' },
  { id: 'market', icon: '📊', text: '行情中心' },
  { id: 'backtest', icon: '📈', text: '策略回测' },
] as const;
</script>

<template>
  <aside class="sidebar">
    <div class="logo-section">
      <div class="logo">
        <span class="logo-icon">📈</span>
        <span class="logo-text">FinGent</span>
      </div>
      <p class="logo-subtitle">AI 智能投研助手</p>
    </div>
    
    <!-- 主导航 -->
    <nav class="main-nav">
      <button
        v-for="item in navItems"
        :key="item.id"
        class="nav-btn"
        :class="{ active: currentView === item.id }"
        @click="emit('switch-view', item.id)"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-text">{{ item.text }}</span>
      </button>
    </nav>
    
    <!-- 聊天模式下的新对话按钮 -->
    <button 
      v-if="currentView === 'chat'"
      class="new-chat" 
      @click="emit('new-chat')"
    >
      <span class="plus-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 5v14M5 12h14"/>
        </svg>
      </span>
      <span>新对话</span>
    </button>
    
    <div v-if="currentView === 'chat'" class="quick-actions">
      <h3>快捷提问</h3>
      <div class="actions-list">
        <button 
          v-for="action in quickActions" 
          :key="action.text"
          @click="emit('quick-input', action.prompt)"
        >
          <span class="action-icon">{{ action.icon }}</span>
          <span class="action-text">{{ action.text }}</span>
        </button>
      </div>
    </div>
    
    <div class="footer">
      <div class="footer-icon">⚡</div>
      <div class="footer-text">
        <p>AI 分析仅供参考</p>
        <p>投资有风险，入市需谨慎</p>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 280px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%);
  color: #1e293b;
  display: flex;
  flex-direction: column;
  padding: 1.5rem;
  border-right: 1px solid rgba(0, 0, 0, 0.06);
  backdrop-filter: blur(20px);
}

.logo-section {
  margin-bottom: 1.5rem;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.5rem;
  font-weight: 800;
  margin-bottom: 0.375rem;
}

.logo-icon {
  font-size: 1.75rem;
}

.logo-text {
  background: linear-gradient(to right, #60a5fa, #34d399);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.02em;
}

.logo-subtitle {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.45);
  margin: 0;
  padding-left: 2.5rem;
  letter-spacing: 0.05em;
}

/* 主导航样式 */
.main-nav {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
}

.nav-btn {
  display: flex;
  align-items: center;
  gap: 0.875rem;
  padding: 0.875rem 1rem;
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 0.75rem;
  color: rgba(0, 0, 0, 0.65);
  font-size: 0.9375rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.nav-btn:hover {
  background: rgba(0, 0, 0, 0.05);
  color: #1e293b;
  transform: translateX(2px);
}

.nav-btn.active {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(124, 58, 237, 0.08) 100%);
  border-color: rgba(139, 92, 246, 0.35);
  color: #7c3aed;
}

.nav-icon {
  font-size: 1.25rem;
  width: 1.5rem;
  text-align: center;
}

.new-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.875rem;
  background: linear-gradient(135deg, #3b82f6, #06b6d4);
  color: white;
  border: none;
  border-radius: 0.875rem;
  font-size: 0.9375rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 1.75rem;
  box-shadow: 0 4px 20px -5px rgba(59, 130, 246, 0.5);
}

.new-chat:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 30px -8px rgba(59, 130, 246, 0.6);
}

.plus-icon {
  width: 1.25rem;
  height: 1.25rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.plus-icon svg {
  width: 100%;
  height: 100%;
}

.quick-actions {
  flex: 1;
}

.quick-actions h3 {
  font-size: 0.6875rem;
  text-transform: uppercase;
  color: rgba(0, 0, 0, 0.4);
  margin-bottom: 0.875rem;
  letter-spacing: 0.12em;
  font-weight: 700;
  padding-left: 0.5rem;
}

.actions-list {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.actions-list button {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: transparent;
  color: rgba(0, 0, 0, 0.65);
  border: none;
  border-radius: 0.625rem;
  text-align: left;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
}

.actions-list button:hover {
  background: rgba(0, 0, 0, 0.04);
  color: #1e293b;
}

.action-icon {
  font-size: 1.125rem;
  width: 1.5rem;
  text-align: center;
}

.action-text {
  font-weight: 500;
}

.footer {
  margin-top: auto;
  padding-top: 1.25rem;
  border-top: 1px solid rgba(0, 0, 0, 0.06);
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.footer-icon {
  font-size: 1.25rem;
  opacity: 0.8;
}

.footer-text {
  flex: 1;
}

.footer-text p {
  font-size: 0.6875rem;
  color: rgba(0, 0, 0, 0.4);
  margin: 0;
  line-height: 1.5;
}
</style>
