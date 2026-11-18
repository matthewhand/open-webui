<script lang="ts">
	import { getPageContext } from '$lib/utils/page';
	const { goto } = getPageContext();

	import { config, user } from '$lib/stores';
	import { onMount } from 'svelte';

	onMount(() => {
		if ($user?.role !== 'admin') {
			if ($user?.permissions?.workspace?.models) {
				goto('/workspace/models', { replace: true });
			} else if ($user?.permissions?.workspace?.knowledge) {
				goto('/workspace/knowledge', { replace: true });
			} else if ($user?.permissions?.workspace?.prompts) {
				goto('/workspace/prompts', { replace: true });
			} else if ($config?.features?.enable_tools && $user?.permissions?.workspace?.tools) {
				goto('/workspace/tools', { replace: true });
			} else if ($user?.permissions?.workspace?.skills) {
				goto('/workspace/skills', { replace: true });
			} else {
				goto('/', { replace: true });
			}
		} else {
			goto('/workspace/models', { replace: true });
		}
	});
</script>
