/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['DM Sans', 'Helvetica Neue', 'Arial', 'sans-serif'],
        display: ['Outfit', 'Helvetica Neue', 'Arial', 'sans-serif'],
        mid: ['Poppins', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      colors: {
        'notebook': {
          50:  '#f8f8f8',
          100: '#f2f3f5',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#8e8e93',
          500: '#5f5f5f',
          600: '#45515e',
          700: '#2d3a45',
          800: '#181e25',
          900: '#222222',
        },
        'brand': {
          DEFAULT: '#1456f0',
          hover:   '#2563eb',
          pressed: '#1d4ed8',
          dark:    '#181e25',
        },
      },
      boxShadow: {
        'brand-glow': 'rgba(44, 30, 116, 0.16) 0px 0px 15px',
        'card':       'rgba(0, 0, 0, 0.08) 0px 4px 6px',
        'elevated':   'rgba(36, 36, 36, 0.08) 0px 12px 16px -4px',
      },
    },
  },
  plugins: [],
}
