<script lang="ts">
	import { toast } from 'svelte-sonner';

	import { page } from '$lib/stores/currentPage';

	$: error = $page.url.searchParams.get('error');
	$: if (error) toast.error(error);
</script>
