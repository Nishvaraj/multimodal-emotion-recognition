/*
  PostCSS configuration for the Multi-Modal Emotion Recognition frontend build.

  This module defines the CSS transformation pipeline used by the React toolchain.
*/

// --- Build Pipeline Configuration ---
module.exports = {
  // Standard PostCSS pipeline for the React build.
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
