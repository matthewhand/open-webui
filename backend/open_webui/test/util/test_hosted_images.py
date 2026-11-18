import asyncio
import importlib.util
import io
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image, UnidentifiedImageError

module_path = Path(__file__).parents[2] / 'utils' / 'images' / 'hosted.py'
spec = importlib.util.spec_from_file_location('hosted_image_providers', module_path)
hosted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosted)


def provider_response(data=None, content_type='application/json', chunks=()):
    response = MagicMock()
    response.content_type = content_type
    response.json = AsyncMock(return_value=data)

    async def iter_chunks(_):
        for chunk in chunks:
            yield chunk

    response.content.iter_chunked.side_effect = iter_chunks
    context = MagicMock()
    context.__aenter__.return_value = response
    return context


class HostedImageTests(IsolatedAsyncioTestCase):
    async def generate(self, session, engine, model, params=None, count=1):
        return await hosted.generate_hosted_images(
            session,
            {
                'type': engine,
                'url': hosted.HOSTED_IMAGE_PROVIDERS[engine],
                'key': 'test-key',
                'config': {'params': params or {}},
            },
            model,
            'a landscape',
            (640, 480),
            count,
            25,
            'text',
            True,
            AsyncMock(),
        )

    async def test_together_sends_model_dimensions_and_output_parameters(self):
        session = MagicMock()
        session.post.return_value = provider_response(
            {'data': [{'b64_json': 'base64'}, {'url': 'https://images.example/test'}]}
        )
        result = await self.generate(session, 'togetherai', 'test/model', {'response_format': 'b64_json'}, 2)
        self.assertEqual(result, ['base64', 'https://images.example/test'])
        request = session.post.call_args
        self.assertEqual(request.args[0], 'https://api.together.ai/v1/images/generations')
        self.assertEqual(
            request.kwargs['json'],
            {
                'model': 'test/model',
                'prompt': 'a landscape',
                'width': 640,
                'height': 480,
                'steps': 25,
                'n': 2,
                'negative_prompt': 'text',
                'response_format': 'b64_json',
            },
        )
        self.assertEqual(request.kwargs['headers'], {'Authorization': 'Bearer test-key'})
        self.assertFalse(request.kwargs['allow_redirects'])

    async def test_hugging_face_reads_image_stream(self):
        session = MagicMock()
        session.post.return_value = provider_response(content_type='image/png', chunks=(b'img', b'data'))
        result = await self.generate(session, 'huggingface', 'test/model')
        self.assertEqual(result, [(b'imgdata', 'image/png')])
        self.assertEqual(session.post.call_args.args[0], 'https://router.huggingface.co/hf-inference/models/test/model')
        self.assertEqual(session.post.call_args.kwargs['json']['parameters']['num_inference_steps'], 25)
        self.assertFalse(session.post.call_args.kwargs['allow_redirects'])

    async def test_replicate_polls_the_configured_origin(self):
        session = MagicMock()
        session.post.return_value = provider_response(
            {'status': 'processing', 'id': 'valid-prediction', 'urls': {'get': 'https://other.example/poll'}}
        )
        session.get.return_value = provider_response({'status': 'succeeded', 'output': 'https://images.example/test'})
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            result = await self.generate(session, 'replicate', 'test/model')
        self.assertEqual(result, ['https://images.example/test'])
        self.assertEqual(session.post.call_args.args[0], 'https://api.replicate.com/v1/models/test/model/predictions')
        self.assertEqual(session.get.call_args.args[0], 'https://api.replicate.com/v1/predictions/valid-prediction')
        self.assertFalse(session.get.call_args.kwargs['allow_redirects'])

    async def test_invalid_model_fails_before_request(self):
        session = MagicMock()
        with self.assertRaisesRegex(ValueError, 'Enter the Replicate model'):
            await self.generate(session, 'replicate', '/absolute/path')
        session.post.assert_not_called()

    def test_raster_validation_reads_the_actual_format(self):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2)).save(buffer, format='PNG')
        self.assertEqual(hosted.validate_hosted_image(buffer.getvalue()), 'image/png')

    def test_raster_validation_rejects_non_images(self):
        with self.assertRaises(UnidentifiedImageError):
            hosted.validate_hosted_image(b'not a valid image')
