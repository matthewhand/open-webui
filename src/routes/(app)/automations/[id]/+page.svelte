<script lang="ts">
	import Automation from '$lib/components/automations/Automation.svelte';
	import { page } from '$lib/stores/currentPage';
</script>

<Automation id={$page.params.id} />
