<template>
  <div class="loading-message">
    <div class="ai-avatar">
      <span class="avatar-icon">🤖</span>
      <div class="avatar-pulse"></div>
    </div>
    <div class="content">
      <div class="loading-bubble">
        <div class="thinking-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <span class="loading-text">AI 正在分析...</span>
      </div>
      <p class="loading-subtitle">正在查询数据并生成报告</p>
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
}

.loading-bubble {
  display: flex;
  align-items: center;
  gap: 0.875rem;
  padding: 1rem 1.25rem;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(0, 0, 0, 0.08);
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
  color: rgba(0, 0, 0, 0.65);
  font-weight: 500;
}

.loading-subtitle {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.4);
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
