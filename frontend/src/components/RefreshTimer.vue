<template>
  <!-- 纯逻辑组件，不渲染任何 UI -->
</template>

<script setup>
import { onMounted, onBeforeUnmount } from 'vue'

const props = defineProps({
  interval: {
    type: Number,
    default: 30000, // 30秒
  },
})

const emit = defineEmits(['tick'])

let timer = null

onMounted(() => {
  timer = setInterval(() => {
    emit('tick')
  }, props.interval)
})

onBeforeUnmount(() => {
  if (timer) {
    clearInterval(timer)
  }
})
</script>