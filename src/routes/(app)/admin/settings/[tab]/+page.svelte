<script>
	import { goto } from '$app/navigation';
	import { page } from '$lib/stores/currentPage';
	import { onMount } from 'svelte';

	onMount(() => {
		const params = new URLSearchParams($page.url.searchParams);
		params.set('settings', `admin:${$page.params.tab ?? 'general'}`);
		goto(`/?${params.toString()}`, { replace: true });
	});
</script>
