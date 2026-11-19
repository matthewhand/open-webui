<script lang="ts">
	import EditPrompt from '$lib/components/workspace/Prompts/EditPrompt.svelte';
	import { page } from '$lib/stores/currentPage';
</script>

<EditPrompt id={$page.params.id} />
