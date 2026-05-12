<script setup lang="ts">
interface Props {
  threadId: string | null;
}

defineProps<Props>();
</script>

<template>
  <header class="header">
    <div class="header-content">
      <div class="brand">
        <span class="brand-icon">🤖</span>
        <div class="brand-text">
          <h1>FinGent</h1>
          <span class="brand-tagline">智能股票分析</span>
        </div>
      </div>
      <div v-if="threadId" class="thread-badge">
        <span class="thread-label">会话</span>
        <span class="thread-id">{{ threadId.slice(-8) }}</span>
      </div>
    </div>
  </header>
</template>

<style scoped>
.header {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(20px);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 56rem;
  margin: 0 auto;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.brand-icon {
  font-size: 1.5rem;
}

.brand-text {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.brand-text h1 {
  font-size: 1.125rem;
  font-weight: 700;
  color: white;
  margin: 0;
  background: linear-gradient(to right, #60a5fa, #34d399);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.brand-tagline {
  font-size: 0.6875rem;
  color: rgba(0, 0, 0, 0.45);
  letter-spacing: 0.05em;
}

.thread-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.375rem 0.875rem;
  background: rgba(0, 0, 0, 0.04);
  border-radius: 9999px;
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.thread-label {
  font-size: 0.625rem;
  text-transform: uppercase;
  color: rgba(0, 0, 0, 0.45);
  letter-spacing: 0.1em;
  font-weight: 600;
}

.thread-id {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.65);
  font-family: monospace;
  letter-spacing: 0.05em;
}
</style>
