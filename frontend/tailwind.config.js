/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0f',
        surface: 'rgba(255, 255, 255, 0.05)',
        surfaceHover: 'rgba(255, 255, 255, 0.08)',
        border: 'rgba(255, 255, 255, 0.1)',
        primary: '#3b82f6',
        primaryHover: '#60a5fa',
        text: '#f8fafc',
        textMuted: '#94a3b8',
        success: '#10b981',
        error: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'float': '0 20px 40px rgba(0,0,0,0.2)',
        'float-heavy': '0 30px 60px rgba(0,0,0,0.4)',
      }
    },
  },
  plugins: [],
}
