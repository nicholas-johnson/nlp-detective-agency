import path from 'path';
import { fileURLToPath } from 'url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const slidesDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(slidesDir, '../..');
const slideDeckEntry = path.resolve(repoRoot, 'slide-deck/src/index.js');

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      'slide-deck': slideDeckEntry,
    },
  },
  optimizeDeps: {
    // Always compile slide-deck from source so HMR picks up shared component changes.
    exclude: ['slide-deck'],
  },
  server: {
    fs: {
      allow: [repoRoot],
    },
  },
});
