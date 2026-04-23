<template>
  <div class="min-h-screen flex flex-col">

    <!-- ── MAIN: two panels ──────────────────────────────────── -->
    <div class="flex flex-1">

      <!-- ── LEFT PANEL (showcase) ─────────────────────────── -->
      <div class="hidden lg:flex lg:w-[60%] bg-[#f2f3f5] flex-col items-center justify-center px-12 py-14 relative overflow-hidden sticky top-0 self-start h-screen">

        <!-- Decorative blobs -->
        <div class="absolute top-[-80px] right-[-80px] w-72 h-72 rounded-full opacity-20"
             style="background: radial-gradient(circle, #1456f0, #a855f7);"></div>
        <div class="absolute bottom-[-60px] left-[-60px] w-56 h-56 rounded-full opacity-15"
             style="background: radial-gradient(circle, #a855f7, #1456f0);"></div>

        <!-- Content -->
        <div class="relative z-10 w-full max-w-sm flex flex-col items-center gap-10">

          <!-- Brand identity -->
          <div class="flex flex-col items-center gap-3">
            <img src="/O_logo.png" alt="OpenLab" class="w-20 h-20 object-contain" />
            <span class="font-display text-3xl font-semibold text-[#222222] tracking-tight">OpenLab</span>
            <p class="text-sm text-[#45515e] text-center leading-relaxed">
              Your AI-powered research workspace.<br>From paper to insight in minutes.
            </p>
          </div>

          <!-- Feature cards -->
          <div class="w-full space-y-3">

            <div class="bg-white rounded-2xl p-4 flex items-start gap-4"
                 style="box-shadow: rgba(44,30,116,0.10) 0px 4px 16px;">
              <div class="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                   style="background: linear-gradient(135deg, #1456f0, #6366f1);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
                </svg>
              </div>
              <div>
                <p class="text-sm font-semibold text-[#222222] mb-0.5">Paper Analysis</p>
                <p class="text-xs text-[#45515e] leading-relaxed">Ask questions across multiple research papers with cited answers.</p>
              </div>
            </div>

            <div class="bg-white rounded-2xl p-4 flex items-start gap-4"
                 style="box-shadow: rgba(44,30,116,0.10) 0px 4px 16px;">
              <div class="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                   style="background: linear-gradient(135deg, #6366f1, #a855f7);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
                </svg>
              </div>
              <div>
                <p class="text-sm font-semibold text-[#222222] mb-0.5">Code Generation</p>
                <p class="text-xs text-[#45515e] leading-relaxed">Reproduce methods and algorithms directly from any paper.</p>
              </div>
            </div>

            <div class="bg-white rounded-2xl p-4 flex items-start gap-4"
                 style="box-shadow: rgba(44,30,116,0.10) 0px 4px 16px;">
              <div class="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                   style="background: linear-gradient(135deg, #a855f7, #ec4899);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
                </svg>
              </div>
              <div>
                <p class="text-sm font-semibold text-[#222222] mb-0.5">Conference Posters</p>
                <p class="text-xs text-[#45515e] leading-relaxed">Auto-generate a presentation-ready poster from your PDF.</p>
              </div>
            </div>

          </div>
        </div>
      </div>

      <!-- ── RIGHT PANEL (form) ─────────────────────────────── -->
      <div class="flex-1 lg:max-w-[40%] bg-white flex flex-col items-center justify-center px-8 py-10 lg:px-16 lg:py-14">

        <!-- Headline -->
        <div class="mb-10">
          <h1 class="font-display text-[3.25rem] font-medium leading-[1.10] text-[#222222] mb-4">
            Think deeper,<br>publish faster.
          </h1>
          <p class="text-base font-normal leading-[1.5] text-[#45515e]">
            AI-powered paper analysis, code generation,<br class="hidden lg:block" /> and poster creation.
          </p>
        </div>

        <!-- Form area -->
        <div class="w-full max-w-sm">

          <!-- Inline Error -->
          <div v-if="errorMessage" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg">
            <p class="text-sm text-red-600">{{ errorMessage }}</p>
          </div>

          <!-- Pill Tab Toggle -->
          <div class="flex bg-[rgba(0,0,0,0.05)] rounded-full p-1 mb-6">
            <button
              @click="tab = 'login'"
              :class="[
                'flex-1 py-2 text-sm font-medium rounded-full transition-all',
                tab === 'login'
                  ? 'bg-white text-[#18181b] shadow-[rgba(0,0,0,0.08)_0px_4px_6px]'
                  : 'text-[#45515e] hover:text-[#222222]'
              ]"
            >Sign in</button>
            <button
              @click="tab = 'register'"
              :class="[
                'flex-1 py-2 text-sm font-medium rounded-full transition-all',
                tab === 'register'
                  ? 'bg-white text-[#18181b] shadow-[rgba(0,0,0,0.08)_0px_4px_6px]'
                  : 'text-[#45515e] hover:text-[#222222]'
              ]"
            >Create account</button>
          </div>

          <!-- Login Form -->
          <form v-if="tab === 'login'" @submit.prevent="handleLogin" class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-[#222222] mb-1.5">Email</label>
              <input
                v-model="loginEmail"
                type="email"
                required
                placeholder="you@example.com"
                class="w-full px-4 py-2.5 border border-[#e5e7eb] bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:ring-[#1456f0] focus:border-transparent transition-shadow"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-[#222222] mb-1.5">Password</label>
              <input
                v-model="loginPassword"
                type="password"
                required
                placeholder="••••••••"
                class="w-full px-4 py-2.5 border border-[#e5e7eb] bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:ring-[#1456f0] focus:border-transparent transition-shadow"
              />
            </div>
            <button
              type="submit"
              :disabled="isLoading"
              class="w-full px-5 py-[11px] bg-[#181e25] text-white rounded-lg text-sm font-semibold hover:bg-[#2d3a45] transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-2"
            >{{ isLoading ? 'Signing in…' : 'Sign in' }}</button>
          </form>

          <!-- Register Form -->
          <form v-else @submit.prevent="handleRegister" class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-[#222222] mb-1.5">Username</label>
              <input
                v-model="regUsername"
                type="text"
                required
                placeholder="johndoe"
                class="w-full px-4 py-2.5 border border-[#e5e7eb] bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:ring-[#1456f0] focus:border-transparent transition-shadow"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-[#222222] mb-1.5">Email</label>
              <input
                v-model="regEmail"
                type="email"
                required
                placeholder="you@example.com"
                class="w-full px-4 py-2.5 border border-[#e5e7eb] bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:ring-[#1456f0] focus:border-transparent transition-shadow"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-[#222222] mb-1.5">Password</label>
              <input
                v-model="regPassword"
                type="password"
                required
                placeholder="••••••••"
                class="w-full px-4 py-2.5 border border-[#e5e7eb] bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:ring-[#1456f0] focus:border-transparent transition-shadow"
              />
            </div>
            <button
              type="submit"
              :disabled="isLoading"
              class="w-full px-5 py-[11px] bg-[#181e25] text-white rounded-lg text-sm font-semibold hover:bg-[#2d3a45] transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-2"
            >{{ isLoading ? 'Creating account…' : 'Create account' }}</button>
          </form>

        </div>
      </div>

    </div>

    <!-- ── FOOTER ──────────────────────────────────────────── -->
    <footer class="w-full bg-[#181e25] py-4 px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
      <div class="flex items-center gap-2">
        <img src="/O_logo.png" alt="OpenLab" class="w-5 h-5 object-contain opacity-60" />
        <span class="text-xs text-white/40">&copy; {{ new Date().getFullYear() }} OpenLab. All rights reserved.</span>
      </div>
      <div class="flex items-center gap-5">
        <a href="#" class="text-xs text-white/40 hover:text-white/70 transition-colors">About</a>
        <a href="#" class="text-xs text-white/40 hover:text-white/70 transition-colors">Documentation</a>
        <a href="#" class="text-xs text-white/40 hover:text-white/70 transition-colors">Privacy</a>
        <a href="#" class="text-xs text-white/40 hover:text-white/70 transition-colors">Terms</a>
        <a href="#" class="text-xs text-white/40 hover:text-white/70 transition-colors">Contact</a>
      </div>
    </footer>

  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const router = useRouter()

const tab = ref('login')
const isLoading = ref(false)
const errorMessage = ref('')

const loginEmail = ref('')
const loginPassword = ref('')

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
