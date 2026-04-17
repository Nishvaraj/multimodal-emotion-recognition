module.exports = {
  // Tailwind scans the React source tree for utility classes.
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  // Custom theme tokens used by the app shell and cards.
  theme: {
    extend: {
      colors: {
        // Gradio-style dark theme colors
        'slate': {
          50: '#f8fafc',
          400: '#94a3b8',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        'purple': {
          300: '#a78bfa',
          400: '#8b5cf6',
          500: '#8b5cf6',
          600: '#7c3aed',
          900: '#4c1d95',
        },
        'indigo': {
          500: '#6366f1',
          600: '#6366f1',
        }
      },
      backgroundImage: {
        'gradient-purple': 'linear-gradient(135deg, #7c3aed, #6366f1)',
      }
    },
  },
  plugins: [],
}

