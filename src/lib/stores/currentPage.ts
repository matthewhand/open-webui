import { browser } from '$app/env';
import { page as route } from '$app/state';
import { toStore, type Readable } from 'svelte/store';

// Bridge Kit's reactive page state to the store subscriptions in legacy components.
// Read properties explicitly so route, query-string, and shallow state changes notify subscribers.
function snapshot() {
	return {
		data: route.data,
		error: route.error,
		form: route.form,
		params: route.params,
		route: route.route,
		state: route.state,
		status: route.status,
		url: route.url
	};
}

let clientPage: Readable<ReturnType<typeof snapshot>> | undefined;

export const page: Readable<ReturnType<typeof snapshot>> = {
	subscribe(run, invalidate) {
		const store = browser ? (clientPage ??= toStore(snapshot)) : toStore(snapshot);
		return store.subscribe(run, invalidate);
	}
};
