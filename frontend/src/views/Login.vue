<template>
  <div class="min-h-screen flex flex-col">

    <!-- ── MAIN: two panels ──────────────────────────────────── -->
    <div class="flex flex-1">

      <!-- ── LEFT PANEL (showcase) ─────────────────────────── -->
      <div
        class="hidden lg:flex lg:w-[60%] bg-[#f2f3f5] flex-col items-center justify-center px-16 py-14 relative overflow-hidden sticky top-0 self-start h-screen"
        @mouseenter="stopTimer"
        @mouseleave="startTimer"
      >
        <!-- Decorative blobs -->
        <div class="absolute top-[-80px] right-[-80px] w-80 h-80 rounded-full opacity-20"
             style="background: radial-gradient(circle, #1456f0, #a855f7);"></div>
        <div class="absolute bottom-[-60px] left-[-60px] w-64 h-64 rounded-full opacity-15"
             style="background: radial-gradient(circle, #a855f7, #1456f0);"></div>

        <!-- Content -->
        <div class="relative z-10 w-full max-w-md flex flex-col items-center gap-10">

          <!-- Brand identity -->
          <div class="flex flex-col items-center gap-3">
            <img src="/OpenLab_logo_lightmode.png" alt="OpenLab" class="w-30 h-16 object-contain" draggable="false"/>
            <p class="text-sm text-[#45515e] text-center leading-relaxed">
              Your AI-powered research workspace.<br>From paper to insight in minutes.
            </p>
          </div>

          <!-- Feature card carousel -->
          <div class="w-full">
            <Transition name="card" mode="out-in">
              <div
                :key="activeCard"
                class="bg-white rounded-2xl overflow-hidden"
                style="box-shadow: rgba(44,30,116,0.13) 0px 10px 40px;"
              >
                <!-- Gradient header -->
                <div
                  class="h-40 flex items-center justify-center relative"
                  :style="{ background: cards[activeCard].gradient }"
                >
                  <div class="absolute -top-4 -right-4 w-36 h-36 rounded-full bg-white/10"></div>
                  <div class="absolute -bottom-3 -left-3 w-24 h-24 rounded-full bg-white/10"></div>
                  <div class="relative z-10 w-16 h-16 rounded-2xl bg-white/25 flex items-center justify-center">
                    <svg
                      width="28" height="28" viewBox="0 0 24 24"
                      fill="none" stroke="white" stroke-width="1.75"
                      stroke-linecap="round" stroke-linejoin="round"
                      v-html="cards[activeCard].iconHtml"
                    ></svg>
                  </div>
                </div>
                <!-- Text content -->
                <div class="px-6 py-5">
                  <p class="text-base font-bold text-[#222222] mb-1.5">{{ cards[activeCard].title }}</p>
                  <p class="text-sm text-[#45515e] leading-relaxed">{{ cards[activeCard].description }}</p>
                </div>
              </div>
            </Transition>

            <!-- Nav row: prev · dots · next -->
            <div class="flex items-center justify-center gap-4 mt-5">
              <button
                @click="prev"
                class="w-8 h-8 flex items-center justify-center rounded-full text-[#45515e] hover:text-[#1456f0] hover:bg-white transition-all"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="15 18 9 12 15 6"/>
                </svg>
              </button>

              <div class="flex items-center gap-2">
                <button
                  v-for="(_, i) in cards"
                  :key="i"
                  @click="goTo(i)"
                  class="rounded-full transition-all duration-300"
                  :class="i === activeCard ? 'w-5 h-2 bg-[#1456f0]' : 'w-2 h-2 bg-[#c7cdd5] hover:bg-[#8e8e93]'"
                ></button>
              </div>

              <button
                @click="next"
                class="w-8 h-8 flex items-center justify-center rounded-full text-[#45515e] hover:text-[#1456f0] hover:bg-white transition-all"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="9 18 15 12 9 6"/>
                </svg>
              </button>
            </div>
          </div>

        </div>
      </div>

      <!-- ── RIGHT PANEL (form) ─────────────────────────────── -->
      <!--
        overflow-y-auto: register form (3 fields) is taller than login; on shorter
        viewports content overflowed onto the dark footer. The my-auto inner wrapper
        keeps content vertically centered when there is space, and the scroll kicks in
        only when there isn't.
      -->
      <div class="flex-1 lg:max-w-[40%] bg-white flex flex-col items-center overflow-y-auto px-8 py-10 lg:px-16 lg:py-14">

        <div class="w-full max-w-sm my-auto flex flex-col">

          <!-- Headline — transitions with tab switch -->
          <Transition name="panel" mode="out-in">
            <div :key="tab" class="mb-10">
              <h1 class="font-display text-[3.25rem] font-medium leading-[1.10] text-[#222222] mb-4">
                {{ headlines[tab].line1 }}<br>{{ headlines[tab].line2 }}
              </h1>
              <p class="text-base font-normal leading-[1.5] text-[#45515e]">
                {{ headlines[tab].body }}
              </p>
            </div>
          </Transition>

          <!-- API error banner (lives outside the keyed transition so it persists) -->
          <div v-if="errorMessage" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg">
            <p class="text-sm text-red-600">{{ errorMessage }}</p>
          </div>

          <Transition name="panel" mode="out-in">
            <div :key="tab">

              <!-- Pill Tab Toggle — sliding indicator -->
              <div class="flex bg-[rgba(0,0,0,0.05)] rounded-full p-1 mb-6 relative overflow-hidden">
                <div
                  class="absolute inset-y-1 left-1 bg-white rounded-full shadow-[rgba(0,0,0,0.08)_0px_4px_6px] pointer-events-none transition-transform duration-200 ease-in-out"
                  :style="{ width: 'calc(50% - 4px)', transform: tab === 'register' ? 'translateX(100%)' : 'translateX(0)' }"
                ></div>
                <button
                  @click="switchTab('login')"
                  class="flex-1 py-2 text-sm font-medium rounded-full relative z-10 transition-colors duration-200"
                  :class="tab === 'login' ? 'text-[#18181b]' : 'text-[#45515e] hover:text-[#222222]'"
                >Sign in</button>
                <button
                  @click="switchTab('register')"
                  class="flex-1 py-2 text-sm font-medium rounded-full relative z-10 transition-colors duration-200"
                  :class="tab === 'register' ? 'text-[#18181b]' : 'text-[#45515e] hover:text-[#222222]'"
                >Create account</button>
              </div>

              <!-- Login fields -->
              <form v-if="tab === 'login'" @submit.prevent="handleLogin" class="space-y-4">
                <div>
                  <label class="block text-sm font-medium text-[#222222] mb-1.5">Email</label>
                  <input
                    v-model="loginEmail"
                    type="text"
                    placeholder="you@example.com"
                    @input="clearFieldError('loginEmail')"
                    :class="[
                      'w-full px-4 py-2.5 border bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:border-transparent transition-shadow',
                      fieldErrors.loginEmail ? 'border-red-400 focus:ring-red-300' : 'border-[#e5e7eb] focus:ring-[#1456f0]'
                    ]"
                  />
                  <Transition name="err"><p v-if="fieldErrors.loginEmail" class="text-xs text-red-500 mt-1">{{ fieldErrors.loginEmail }}</p></Transition>
                </div>
                <div>
                  <label class="block text-sm font-medium text-[#222222] mb-1.5">Password</label>
                  <input
                    v-model="loginPassword"
                    type="password"
                    placeholder="••••••••"
                    @input="clearFieldError('loginPassword')"
                    :class="[
                      'w-full px-4 py-2.5 border bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:border-transparent transition-shadow',
                      fieldErrors.loginPassword ? 'border-red-400 focus:ring-red-300' : 'border-[#e5e7eb] focus:ring-[#1456f0]'
                    ]"
                  />
                  <Transition name="err"><p v-if="fieldErrors.loginPassword" class="text-xs text-red-500 mt-1">{{ fieldErrors.loginPassword }}</p></Transition>
                </div>
                <button
                  type="submit"
                  :disabled="isLoading"
                  class="w-full px-5 py-[11px] bg-[#181e25] text-white rounded-lg text-sm font-semibold hover:bg-[#2d3a45] transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-2"
                >{{ isLoading ? 'Signing in…' : 'Sign in' }}</button>
              </form>

              <!-- Register fields -->
              <form v-else @submit.prevent="handleRegister" class="space-y-4">
                <div>
                  <label class="block text-sm font-medium text-[#222222] mb-1.5">Username</label>
                  <input
                    v-model="regUsername"
                    type="text"
                    placeholder="johndoe"
                    @input="clearFieldError('regUsername')"
                    :class="[
                      'w-full px-4 py-2.5 border bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:border-transparent transition-shadow',
                      fieldErrors.regUsername ? 'border-red-400 focus:ring-red-300' : 'border-[#e5e7eb] focus:ring-[#1456f0]'
                    ]"
                  />
                  <Transition name="err"><p v-if="fieldErrors.regUsername" class="text-xs text-red-500 mt-1">{{ fieldErrors.regUsername }}</p></Transition>
                </div>
                <div>
                  <label class="block text-sm font-medium text-[#222222] mb-1.5">Email</label>
                  <input
                    v-model="regEmail"
                    type="text"
                    placeholder="you@example.com"
                    @input="clearFieldError('regEmail')"
                    :class="[
                      'w-full px-4 py-2.5 border bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:border-transparent transition-shadow',
                      fieldErrors.regEmail ? 'border-red-400 focus:ring-red-300' : 'border-[#e5e7eb] focus:ring-[#1456f0]'
                    ]"
                  />
                  <Transition name="err"><p v-if="fieldErrors.regEmail" class="text-xs text-red-500 mt-1">{{ fieldErrors.regEmail }}</p></Transition>
                </div>
                <div>
                  <label class="block text-sm font-medium text-[#222222] mb-1.5">Password</label>
                  <input
                    v-model="regPassword"
                    type="password"
                    placeholder="••••••••"
                    @input="clearFieldError('regPassword')"
                    :class="[
                      'w-full px-4 py-2.5 border bg-white rounded-lg text-sm text-[#222222] placeholder-[#8e8e93] focus:outline-none focus:ring-2 focus:border-transparent transition-shadow',
                      fieldErrors.regPassword ? 'border-red-400 focus:ring-red-300' : 'border-[#e5e7eb] focus:ring-[#1456f0]'
                    ]"
                  />
                  <Transition name="err"><p v-if="fieldErrors.regPassword" class="text-xs text-red-500 mt-1">{{ fieldErrors.regPassword }}</p></Transition>
                </div>
                <button
                  type="submit"
                  :disabled="isLoading"
                  class="w-full px-5 py-[11px] bg-[#181e25] text-white rounded-lg text-sm font-semibold hover:bg-[#2d3a45] transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-2"
                >{{ isLoading ? 'Creating account…' : 'Create account' }}</button>
              </form>

            </div>
          </Transition>

        </div>
      </div>

    </div>

    <!-- ── FOOTER ──────────────────────────────────────────── -->
    <footer class="w-full bg-[#181e25] py-4 px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
      <div class="flex items-center gap-2">
        <img src="/O.png" alt="OpenLab" class="w-5 h-5 object-contain opacity-60" />
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
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const router = useRouter()

// ── Carousel ─────────────────────────────────────────────────────

const cards = [
  {
    title: 'Paper Analysis',
    description: 'Ask questions across multiple research papers and get cited, AI-powered answers instantly.',
    gradient: 'linear-gradient(135deg, #1456f0, #6366f1)',
    iconHtml: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
  },
  {
    title: 'Code Generation',
    description: 'Turn any research paper into working code. Reproduce methods and experiments in seconds.',
    gradient: 'linear-gradient(135deg, #6366f1, #a855f7)',
    iconHtml: '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
  },
  {
    title: 'Conference Posters',
    description: 'Auto-generate a polished, presentation-ready poster directly from your PDF paper.',
    gradient: 'linear-gradient(135deg, #a855f7, #ec4899)',
    iconHtml: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>',
  },
  {
    title: 'Smart Notebooks',
    description: 'Organize papers into research notebooks and instantly search across your entire library.',
    gradient: 'linear-gradient(135deg, #0ea5e9, #1456f0)',
    iconHtml: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
  },
  {
    title: 'Semantic Search',
    description: 'Search by concept, not keywords. Find exactly what you need across all your uploaded papers.',
    gradient: 'linear-gradient(135deg, #10b981, #0ea5e9)',
    iconHtml: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
  },
]

const activeCard = ref(0)
const intervalId = ref(null)

function goTo(i) {
  activeCard.value = i
  stopTimer()
  startTimer()
}

function next() { goTo((activeCard.value + 1) % cards.length) }
function prev() { goTo((activeCard.value - 1 + cards.length) % cards.length) }

function startTimer() {
  intervalId.value = setInterval(() => {
    activeCard.value = (activeCard.value + 1) % cards.length
  }, 3500)
}

function stopTimer() {
  clearInterval(intervalId.value)
  intervalId.value = null
}

onMounted(startTimer)
onUnmounted(stopTimer)

// ── Headlines (distinct per tab so the transition is meaningful) ──

const headlines = {
  login: {
    line1: 'Think deeper,',
    line2: 'publish faster.',
    body: 'AI-powered paper analysis, code generation, and poster creation.',
  },
  register: {
    line1: 'Your research,',
    line2: 'supercharged.',
    body: 'Join researchers using AI to read, understand, and publish faster.',
  },
}

// ── Form state ────────────────────────────────────────────────────

const tab = ref('login')
const isLoading = ref(false)
const errorMessage = ref('')

const loginEmail = ref('')
const loginPassword = ref('')

const regUsername = ref('')
const regEmail = ref('')
const regPassword = ref('')

const fieldErrors = reactive({
  loginEmail: '',
  loginPassword: '',
  regUsername: '',
  regEmail: '',
  regPassword: '',
})

function clearFieldError(field) {
  fieldErrors[field] = ''
}

function switchTab(newTab) {
  tab.value = newTab
  errorMessage.value = ''
  Object.keys(fieldErrors).forEach(k => { fieldErrors[k] = '' })
}

// ── Validation ────────────────────────────────────────────────────

const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function validateLogin() {
  let valid = true
  if (!loginEmail.value.trim()) {
    fieldErrors.loginEmail = 'Required'
    valid = false
  } else if (!emailRe.test(loginEmail.value.trim())) {
    fieldErrors.loginEmail = 'Enter a valid email address'
    valid = false
  }
  if (!loginPassword.value) {
    fieldErrors.loginPassword = 'Required'
    valid = false
  }
  return valid
}

function validateRegister() {
  let valid = true
  if (!regUsername.value.trim()) {
    fieldErrors.regUsername = 'Required'
    valid = false
  }
  if (!regEmail.value.trim()) {
    fieldErrors.regEmail = 'Required'
    valid = false
  } else if (!emailRe.test(regEmail.value.trim())) {
    fieldErrors.regEmail = 'Enter a valid email address'
    valid = false
  }
  if (!regPassword.value) {
    fieldErrors.regPassword = 'Required'
    valid = false
  } else if (regPassword.value.length < 8) {
    fieldErrors.regPassword = 'Password must be at least 8 characters'
    valid = false
  }
  return valid
}

// ── Handlers ──────────────────────────────────────────────────────

async function handleLogin() {
  errorMessage.value = ''
  if (!validateLogin()) return
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
  if (!validateRegister()) return
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

<style scoped>
img {
  -webkit-user-drag: none; /* Chặn kéo ảnh trên Chrome, Safari, Edge */
  pointer-events: none; /* Không cho tương tác chuột */
  user-select: none;    /* Không cho chọn */
}

.panel-enter-active,
.panel-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.panel-enter-from {
  opacity: 0;
  transform: translateY(8px);
}
.panel-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

.card-enter-active,
.card-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}
.card-enter-from {
  opacity: 0;
  transform: translateY(10px);
}
.card-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

.err-enter-active {
  transition: opacity 0.15s ease, transform 0.15s ease;
}
.err-leave-active {
  transition: opacity 0.1s ease;
}
.err-enter-from {
  opacity: 0;
  transform: translateY(-4px);
}
.err-leave-to {
  opacity: 0;
}
</style>
