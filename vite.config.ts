import fs from 'node:fs';
import adapter from '@sveltejs/adapter-static';
import tailwindcss from '@tailwindcss/vite';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { sveltekit } from '@sveltejs/kit/vite';
import { defaultClientConditions, defineConfig } from 'vite';
import { resolveBuildInfo } from './scripts/build-info.js';

import { viteStaticCopy } from 'vite-plugin-static-copy';

const backendTarget = process.env.WEBUI_BACKEND_URL || 'http://localhost:8080';
const buildInfo = resolveBuildInfo();
const version = JSON.parse(
	fs.readFileSync(new URL('./package.json', import.meta.url), 'utf8')
).version;

export default defineConfig({
	resolve: {
		conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions]
	},
	plugins: [
		tailwindcss(),
		sveltekit({
			adapter: adapter({ pages: 'build', assets: 'build', fallback: 'index.html' }),
			alias: { $lib: 'src/lib' },
			preprocess: vitePreprocess(),
			version: { name: buildInfo.hash || version || Date.now().toString(), pollInterval: 60000 },
			onwarn: (warning, handler) => {
				if (warning.code !== 'css-unused-selector') handler(warning);
			}
		}),
		viteStaticCopy({
			targets: [
				{ src: 'node_modules/chart.js/dist/chart.umd.min.js', dest: 'visualizations' },
				{ src: 'node_modules/dompurify/dist/purify.min.js', dest: 'visualizations' },
				{ src: 'node_modules/chart.js/LICENSE.md', dest: 'licenses/visualizations/chartjs' },
				{ src: 'node_modules/dompurify/LICENSE', dest: 'licenses/visualizations/dompurify' },
				{
					src: 'src/lib/encryption/vendor/THIRD_PARTY_LICENSES.txt',
					dest: 'licenses/encryption'
				},
				{
					src: ['node_modules/@novnc/novnc/docs/LICENSE*', 'node_modules/@novnc/novnc/AUTHORS'],
					dest: 'licenses/novnc'
				},
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',
					dest: 'wasm'
				},
				{
					src: 'node_modules/pdfjs-dist/wasm/*',
					dest: 'wasm'
				}
			].map((target) => ({ ...target, rename: { stripBase: true } }))
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(buildInfo.hash),
		APP_BUILD_CHANNEL: JSON.stringify(buildInfo.channel)
	},
	build: {
		target: 'es2022',
		sourcemap: true,
		rolldownOptions: {
			treeshake: {
				manualPureFunctions:
					process.env.ENV === 'dev' ? [] : ['console.log', 'console.debug', 'console.error']
			}
		}
	},
	optimizeDeps: {
		rolldownOptions: { transform: { target: 'es2022' } }
	},
	server: {
		proxy: {
			'/api': {
				target: backendTarget,
				changeOrigin: true,
				ws: true
			},
			'/ollama': {
				target: backendTarget,
				changeOrigin: true
			},
			'/openai': {
				target: backendTarget,
				changeOrigin: true
			},
			'/oauth': {
				target: backendTarget,
				changeOrigin: true
			},
			'/ws': {
				target: backendTarget,
				changeOrigin: true,
				ws: true
			}
		}
	},
	worker: {
		format: 'es'
	}
});
