<template>
  <div id="app" class="h-screen w-screen overflow-hidden bg-white">
    <router-view />
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useAppStore } from './stores/app'

const store = useAppStore()

// Fallback initializer — router guard is the primary path, this handles direct page loads
onMounted(async () => {
  if (localStorage.getItem('token') && !store.user) {
    await store.initApp()
  }
})
</script>
