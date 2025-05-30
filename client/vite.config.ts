import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  // Determine port: use VITE_PORT, else default to 5176
  let port = Number(env.VITE_PORT) || 5176;
  // Explicitly prevent use of port 5173
  if (port === 5173) {
    throw new Error('Port 5173 is forbidden for this project. Please set VITE_PORT to a different value in your .env file.');
  }
  return {
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify('YOUR_API_KEY_HERE'),
      'process.env.OPENAI_API_KEY': JSON.stringify('YOUR_OPENAI_API_KEY_HERE')
    },
    server: {
      port,
      strictPort: true // fail if the port is taken, don't auto-increment
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    },
    optimizeDeps: {
      include: ['openai']
    }
  };
});
