export type ImageCapability = 'image_generation' | 'image_edit';
export type ImageCapabilities = Record<ImageCapability, boolean>;

export const hostedImageProviderUrls = {
	huggingface: 'https://router.huggingface.co/hf-inference',
	replicate: 'https://api.replicate.com/v1',
	togetherai: 'https://api.together.ai/v1'
};

/** What a connection type can do: ComfyUI needs the matching workflow, AUTOMATIC1111 cannot edit. */
export const getSupportedImageCapabilities = (
	type: string,
	config: { workflow?: string; edit_workflow?: string } = {}
): ImageCapabilities => {
	if (type === 'automatic1111' || Object.hasOwn(hostedImageProviderUrls, type)) {
		return { image_generation: true, image_edit: false };
	}
	if (type === 'comfyui') {
		return { image_generation: !!config.workflow, image_edit: !!config.edit_workflow };
	}
	return { image_generation: true, image_edit: true };
};
