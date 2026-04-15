import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Login from '../views/Login.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/login',
    name: 'Login',
    component: Login
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async (to) => {
  if (to.name === 'Login') return true

  const token = localStorage.getItem('token')
  if (!token) return { name: 'Login' }

  // Dynamic import avoids circular dependency (store → router → store)
  const { useAppStore } = await import('../stores/app')
  const store = useAppStore()

  if (!store.user) {
    await store.initApp()
    if (!store.user) {
      localStorage.removeItem('token')
      return { name: 'Login' }
    }
  }

  return true
})

export default router
