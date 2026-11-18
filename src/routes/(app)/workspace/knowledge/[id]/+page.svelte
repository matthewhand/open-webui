<script>
	import { page } from '$lib/stores/currentPage';
	import KnowledgeBase from '$lib/components/workspace/Knowledge/KnowledgeBase.svelte';
</script>

<KnowledgeBase id={$page.params.id} />
