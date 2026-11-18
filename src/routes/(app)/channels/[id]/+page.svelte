<script lang="ts">
	import { page } from '$lib/stores/currentPage';

	import Channel from '$lib/components/channel/Channel.svelte';
</script>

<Channel id={$page.params.id} />
