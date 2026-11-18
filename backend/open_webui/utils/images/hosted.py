"""Text-to-image adapters for the fork's hosted image providers."""

import asyncio
import io
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from aiohttp import ClientResponse, ClientSession, ClientTimeout
from PIL import Image

HOSTED_IMAGE_PROVIDERS = {
    'togetherai': 'https://api.together.ai/v1',
    'huggingface': 'https://router.huggingface.co/hf-inference',
    'replicate': 'https://api.replicate.com/v1',
}
HOSTED_PROVIDER_TIMEOUT = ClientTimeout(total=90)
REPOSITORY_MODEL = re.compile(r'[\w.-]+/[\w.-]+', re.ASCII)
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def validate_hosted_image(image_data: bytes | None) -> str:
    if not image_data or len(image_data) > MAX_IMAGE_BYTES:
        raise ValueError('The generated image is empty or exceeds 32 MiB.')
    with Image.open(io.BytesIO(image_data), formats=('PNG', 'JPEG', 'WEBP')) as image:
        image.verify()
        return Image.MIME[image.format]


async def generate_hosted_images(
    session: ClientSession,
    connection: dict,
    model: str,
    prompt: str,
    dimensions: tuple[int, int],
    count: int,
    steps: int | None,
    negative_prompt: str | None,
    ssl: bool,
    check_response: Callable[[ClientResponse], Awaitable[None]],
) -> list[str | tuple[bytes, str]]:
    """Return URLs, base64 data, or raw image bytes to the normal upload path."""
    if not model or not 1 <= count <= 10:
        raise ValueError('Choose a model and request between 1 and 10 images.')

    engine = connection['type']
    url = connection['url']
    config = connection.get('config') or {}
    parameters = config.get('params') or {}
    width, height = dimensions
    headers = {'Authorization': f'Bearer {connection["key"]}'} if connection['key'] else {}
    options = dict(headers=headers, allow_redirects=False, timeout=HOSTED_PROVIDER_TIMEOUT, ssl=ssl)

    if engine == 'togetherai':
        payload = {'width': width, 'height': height, 'response_format': 'url'}
        if steps is not None:
            payload['steps'] = steps
        if negative_prompt:
            payload['negative_prompt'] = negative_prompt
        payload.update(parameters, model=model, prompt=prompt, n=count)
        async with session.post(f'{url}/images/generations', json=payload, **options) as response:
            await check_response(response)
            result = await response.json()
        images = [entry.get('b64_json') or entry.get('url') for entry in result.get('data') or []]
        return [image for image in images if isinstance(image, str) and image]

    if engine == 'huggingface':
        if urlsplit(url).hostname == 'router.huggingface.co':
            if not REPOSITORY_MODEL.fullmatch(model) or '..' in model:
                raise ValueError('Enter the Hugging Face model as owner/name.')
            url = f'{url}/models/{model}'
        inputs = {'width': width, 'height': height}
        if steps is not None:
            inputs['num_inference_steps'] = steps
        if negative_prompt:
            inputs['negative_prompt'] = negative_prompt
        inputs.update(parameters)
        images = []
        async with asyncio.timeout(90):
            for _ in range(count):
                async with session.post(url, json={'inputs': prompt, 'parameters': inputs}, **options) as response:
                    await check_response(response)
                    content_type = response.content_type
                    if content_type not in {'image/jpeg', 'image/png', 'image/webp'}:
                        raise ValueError('The Hugging Face endpoint did not return a supported image.')
                    payload = bytearray()
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        payload.extend(chunk)
                        if len(payload) > MAX_IMAGE_BYTES:
                            raise ValueError('The generated image exceeds 32 MiB.')
                    images.append((bytes(payload), content_type))
        return images

    repo, separator, version = model.partition(':')
    if not REPOSITORY_MODEL.fullmatch(repo) or '..' in repo:
        raise ValueError('Enter the Replicate model as owner/name or owner/name:version.')
    inputs = {**parameters, 'prompt': prompt}
    if count > 1:
        inputs.setdefault('num_outputs', count)
    if negative_prompt:
        inputs.setdefault('negative_prompt', negative_prompt)
    if separator:
        if not re.fullmatch(r'[a-fA-F0-9]{64}', version):
            raise ValueError('The Replicate version must be a 64-character hexadecimal ID.')
        endpoint = f'{url}/predictions'
        payload = {'version': version, 'input': inputs}
    else:
        endpoint = f'{url}/models/{repo}/predictions'
        payload = {'input': inputs}

    async with asyncio.timeout(90):
        options['headers'] = {**headers, 'Prefer': 'wait=10', 'Cancel-After': '90s'}
        async with session.post(endpoint, json=payload, **options) as response:
            await check_response(response)
            prediction = await response.json()

        options['headers'] = headers
        while prediction.get('status') in {'starting', 'processing'}:
            prediction_id = str(prediction.get('id') or '')
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', prediction_id):
                raise ValueError('Replicate returned an invalid prediction ID.')
            await asyncio.sleep(1)
            async with session.get(f'{url}/predictions/{prediction_id}', **options) as response:
                await check_response(response)
                prediction = await response.json()

    if prediction.get('status') != 'succeeded':
        raise ValueError('The Replicate prediction did not succeed.')
    outputs = prediction.get('output')
    if isinstance(outputs, str):
        outputs = [outputs]
    if not isinstance(outputs, list):
        raise ValueError('The Replicate model did not return an image.')
    return [output for output in outputs if isinstance(output, str) and output.startswith('https://')]
