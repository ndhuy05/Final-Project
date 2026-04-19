<template>
  <div class="min-h-screen bg-white flex items-center justify-center p-4">
    <div class="bg-white rounded-3xl shadow-brand-glow w-full max-w-md p-8 border border-notebook-200">
      <!-- Logo / Title -->
      <div class="flex flex-col items-center mb-8">
        <div class="w-12 h-12 bg-brand rounded-xl flex items-center justify-center mb-3">
          <Brain :size="24" class="text-white" />
        </div>
        <h1 class="font-display text-4xl font-semibold text-notebook-900">Vibe</h1>
        <p class="text-sm text-notebook-600 mt-1">AI-powered research assistant</p>
      </div>

      <!-- Tab Toggle — pill style -->
      <div class="flex bg-notebook-100 rounded-full p-1 mb-6">
        <button
          @click="tab = 'login'"
          :class="[
            'flex-1 py-2 text-sm font-medium rounded-full transition-all',
            tab === 'login'
              ? 'bg-white text-notebook-900 shadow-card'
              : 'text-notebook-600 hover:text-notebook-800'
          ]"
        >
          Sign in
        </button>
        <button
          @click="tab = 'register'"
          :class="[
            'flex-1 py-2 text-sm font-medium rounded-full transition-all',
            tab === 'register'
              ? 'bg-white text-notebook-900 shadow-card'
              : 'text-notebook-600 hover:text-notebook-800'
          ]"
        >
          Create account
        </button>
      </div>

      <!-- Inline Error -->
      <div v-if="errorMessage" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg">
        <p class="text-sm text-red-600">{{ errorMessage }}</p>
      </div>

      <!-- Login Form -->
      <form v-if="tab === 'login'" @submit.prevent="handleLogin" class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-notebook-700 mb-1">Email</label>
          <input
            v-model="loginEmail"
            type="email"
            required
            placeholder="you@example.com"
            class="w-full px-4 py-2.5 border border-notebook-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand focus:border-transparent"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-notebook-700 mb-1">Password</label>
          <input
            v-model="loginPassword"
            type="password"
            required
            placeholder="••••••••"
            class="w-full px-4 py-2.5 border border-notebook-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand focus:border-transparent"
          />
        </div>
        <button
          type="submit"
          :disabled="isLoading"
          class="w-full px-5 py-2.5 bg-notebook-800 text-white rounded-lg hover:bg-notebook-900 transition-colors font-medium text-sm disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {{ isLoading ? 'Signing in…' : 'Sign in' }}
        </button>
      </form>

      <!-- Register Form -->
      <form v-else @submit.prevent="handleRegister" class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-notebook-700 mb-1">Username</label>
          <input
            v-model="regUsername"
            type="text"
            required
            placeholder="johndoe"
            class="w-full px-4 py-2.5 border border-notebook-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand focus:border-transparent"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-notebook-700 mb-1">Email</label>
          <input
            v-model="regEmail"
            type="email"
            required
            placeholder="you@example.com"
            class="w-full px-4 py-2.5 border border-notebook-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand focus:border-transparent"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-notebook-700 mb-1">Password</label>
          <input
            v-model="regPassword"
            type="password"
            required
            placeholder="••••••••"
            class="w-full px-4 py-2.5 border border-notebook-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand focus:border-transparent"
          />
        </div>
        <button
          type="submit"
          :disabled="isLoading"
          class="w-full px-5 py-2.5 bg-notebook-800 text-white rounded-lg hover:bg-notebook-900 transition-colors font-medium text-sm disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {{ isLoading ? 'Creating account…' : 'Create account' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { Brain } from 'lucide-vue-next'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const router = useRouter()

const tab = ref('login')
const isLoading = ref(false)
const errorMessage = ref('')

// Login form fields
const loginEmail = ref('')
const loginPassword = ref('')

// Register form fields
const regUsername = ref('')
const regEmail = ref('')
const regPassword = ref('')

async function handleLogin() {
  errorMessage.value = ''
  isLoading.value = true
  try {
    await store.login(loginEmail.value, loginPassword.value)
    router.push('/')
  } catch (err) {
    errorMessage.value = err?.response?.data?.detail || 'Login failed. Please try again.'
  } finally {
    isLoading.value = false
  }
}

async function handleRegister() {
  errorMessage.value = ''
  isLoading.value = true
  try {
    await store.register(regUsername.value, regEmail.value, regPassword.value)
    router.push('/')
  } catch (err) {
    errorMessage.value = err?.response?.data?.detail || 'Registration failed. Please try again.'
  } finally {
    isLoading.value = false
  }
}
</script>
