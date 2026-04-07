<script setup lang="ts">
interface Props {
  suggestion: string;
}

defineProps<Props>();
</script>

<template>
  <div class="portfolio-suggestion">
    <div class="suggestion-header">
      <span class="icon">💼</span>
      <span class="title">组合配置建议</span>
    </div>
    <p class="text">{{ suggestion }}</p>
  </div>
</template>

<style scoped>
.portfolio-suggestion {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(6, 182, 212, 0.08));
  border-radius: 1rem;
  padding: 1.25rem 1.5rem;
  border: 1px solid rgba(59, 130, 246, 0.2);
  margin-top: 0.5rem;
}

.suggestion-header {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  margin-bottom: 0.75rem;
}

.suggestion-header .icon {
  font-size: 1.25rem;
}

.suggestion-header .title {
  font-weight: 600;
  color: #60a5fa;
  font-size: 0.9375rem;
}

.text {
  color: rgba(255, 255, 255, 0.85);
  font-size: 0.9375rem;
  line-height: 1.75;
  margin: 0;
}
</style>
