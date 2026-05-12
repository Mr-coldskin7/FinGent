<script setup lang="ts">
const emit = defineEmits<{
  'quick-input': [text: string];
  send: [];
}>();

const suggestions = [
  { icon: '📊', text: '贵州茅台', prompt: '分析一下茅台' },
  { icon: '🚀', text: '英伟达走势', prompt: 'NVDA最近怎么样' },
  { icon: '💎', text: '价值投资', prompt: '推荐几只价值股' },
  { icon: '🔥', text: '热点板块', prompt: '科技股还能买吗' },
];

const handleClick = (prompt: string) => {
  emit('quick-input', prompt);
  setTimeout(() => emit('send'), 100);
};
</script>

<template>
  <div class="empty-state">
    <div class="welcome">
      <div class="logo-container">
        <span class="welcome-icon">🤖</span>
        <div class="pulse-ring"></div>
      </div>
      <h1>我是 FinGent</h1>
      <p class="subtitle">你的智能股票分析助手</p>
      <p class="description">基于 AI 技术，为你提供专业、易懂的股票分析与投资建议</p>
    </div>
    <div class="suggestions">
      <div 
        v-for="item in suggestions" 
        :key="item.text"
        class="suggestion-card"
        @click="handleClick(item.prompt)"
      >
        <span class="icon">{{ item.icon }}</span>
        <span class="text">{{ item.text }}</span>
        <svg class="arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M5 12h14M12 5l7 7-7 7"/>
        </svg>
      </div>
    </div>
  </div>
</template>

<style scoped>
.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3rem;
  padding: 2rem;
}

.welcome {
  text-align: center;
}

.logo-container {
  position: relative;
  display: inline-block;
  margin-bottom: 1.5rem;
}

.welcome-icon {
  font-size: 4rem;
  display: block;
  position: relative;
  z-index: 1;
  animation: float 3s ease-in-out infinite;
}

.pulse-ring {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: rgba(59, 130, 246, 0.1);
  animation: pulse 2s ease-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

@keyframes pulse {
  0% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.6;
  }
  100% {
    transform: translate(-50%, -50%) scale(1.5);
    opacity: 0;
  }
}

.welcome h1 {
  font-size: 2.25rem;
  font-weight: 800;
  color: #1e293b;
  margin: 0 0 0.5rem 0;
  background: linear-gradient(to right, #3b82f6, #059669, #3b82f6);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gradient 3s linear infinite;
}

@keyframes gradient {
  0% { background-position: 0% center; }
  100% { background-position: 200% center; }
}

.subtitle {
  font-size: 1.125rem;
  color: rgba(0, 0, 0, 0.55);
  margin: 0 0 0.75rem 0;
  font-weight: 500;
}

.description {
  color: rgba(0, 0, 0, 0.45);
  font-size: 0.875rem;
  margin: 0;
  max-width: 360px;
  line-height: 1.6;
}

.suggestions {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  max-width: 480px;
  width: 100%;
}

.suggestion-card {
  display: flex;
  align-items: center;
  gap: 0.875rem;
  padding: 1rem 1.25rem;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 1rem;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.suggestion-card:hover {
  background: rgba(255, 255, 255, 0.95);
  border-color: rgba(59, 130, 246, 0.35);
  transform: translateY(-3px);
  box-shadow: 0 20px 40px -15px rgba(59, 130, 246, 0.15);
}

.suggestion-card .icon {
  font-size: 1.5rem;
}

.suggestion-card .text {
  flex: 1;
  color: rgba(0, 0, 0, 0.8);
  font-size: 0.9375rem;
  font-weight: 500;
}

.suggestion-card .arrow {
  width: 1rem;
  height: 1rem;
  color: rgba(0, 0, 0, 0.25);
  transition: all 0.2s;
}

.suggestion-card:hover .arrow {
  color: #60a5fa;
  transform: translateX(3px);
}
</style>
