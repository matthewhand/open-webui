<script lang="ts">
	import EncryptedChannel from '$lib/components/channel/EncryptedChannel.svelte';
	import { page } from '$lib/stores/currentPage';
</script>

<EncryptedChannel id={$page.params.id} />
