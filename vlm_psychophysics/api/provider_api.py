"""Model shortcuts and documented API settings (2026-11-13)."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from .hf_images import trusted_download_url

ROUTES = {
    "nvidia-internal": ("https://inference-api.nvidia.com/v1", "NVIDIA_API_KEY", None),
    "nvidia-public": (
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
        "moonshotai/kimi-k2.5",
    ),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "kimi-k3.1-pro"),
    "bigmodel": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "glm-4.6v"),
}

# Gateway routing prefixes are account-specific: resolve them from /models.
# Prefer a documented snapshot, then an exact alias; never pick another model.
PRESETS = {
    'astra': dict(ids=('gpt-6-astra',), delivery='ordered-images', token_field='max_completion_tokens',
                  detail='original', reasoning='low', native=False,
                  reference='https://developers.openai.com/api/docs/models/gpt-6-astra'),
    'opus': dict(ids=('claude-opus-5-5', 'claude-opus-5.5', 'anthropic.claude-opus-5-5',
                      'global.anthropic.claude-opus-5-5', 'us.anthropic.claude-opus-5-5'), delivery='ordered-images',
                 token_field='max_tokens', detail=None, reasoning='low', native=False,
                 api_format='messages',
                 reference='https://platform.claude.com/docs/en/models/opus-5-5/overview'),
    'kimi': dict(ids=('kimi-k3.1-pro',), delivery='native', token_field='max_tokens',
                 detail=None, reasoning='low', native=True,
                 reference='https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k3_1-pro-infer'),
    'qwen': dict(ids=('qwen3.8-max-0902', 'qwen3.8-max-2026-09-02', 'qwen3.8-max'),
                 delivery='video-frames', token_field='max_tokens', detail=None, reasoning=None, native=True, enable_thinking=False,
                 reference='https://www.alibabacloud.com/help/en/model-studio/vision'),
    'qwen-flash': dict(ids=('qwen3.8-flash',), delivery='video-frames', token_field='max_tokens',
                       detail=None, reasoning=None, native=True, enable_thinking=False,
                       reference='https://www.alibabacloud.com/help/en/model-studio/vision'),
    'gemini-flash': dict(ids=('gemini-3.8-flash',), delivery='native', token_field='max_tokens',
                         detail=None, reasoning='low', native=True, native_video_part='image_url',
                         reference='https://aistudio.google.com/docs/openai'),
    'gemini-carbon': dict(ids=('gemini-4-carbon',), delivery='native', token_field='max_tokens',
                         detail=None, reasoning='low', native=True, native_video_part='image_url',
                         reference='https://aistudio.google.com/docs/openai'),
    'gemini-argon': dict(ids=('gemini-4-argon',), delivery='native', token_field='max_tokens',
                         detail=None, reasoning='low', native=True, native_video_part='image_url',
                         reference='https://aistudio.google.com/docs/openai'),
    'gemini-hydrogen': dict(ids=('gemini-4-hydrogen',), delivery='native', token_field='max_tokens',
                            detail=None, reasoning='low', native=True, native_video_part='image_url',
                            reference='https://aistudio.google.com/docs/models/gemini-4-hydrogen'),
}
DIRECT_PROVIDERS = {'opus': 'claude', 'kimi': 'kimi', 'qwen': 'qwen-frames',
                    'qwen-flash': 'qwen-frames', **{n: 'gemini' for n in PRESETS if n.startswith('gemini-')}}
ALIASES = {'astral': 'astra', 'gpt-6-astra': 'astra', 'opus5.5': 'opus', 'opus-5.5': 'opus', 'claude-opus-5-5': 'opus',
           'kimi-k3.1-pro': 'kimi', 'qwen3.8-max': 'qwen', 'qwen3.8-flash': 'qwen-flash',
           'gemini-4-carbon': 'gemini-carbon', 'gemini-4-argon': 'gemini-argon',
           'gemini-3.8-flash': 'gemini-flash'}
ALIASES['gemini-4-hydrogen'] = 'gemini-hydrogen'
ALIASES.update(argon='gemini-argon', carbon='gemini-carbon', hydrogen='gemini-hydrogen')


def preset_name(value):
    value = value.lower().strip()
    return ALIASES.get(value, value)


def model_ids(endpoint, key):
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    request = Request(endpoint + '/models', headers={'Authorization': 'Bearer ' + key})
    try:
        with build_opener(NoRedirect).open(request, timeout=30) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('Model list is too large.')
        data = json.loads(raw)['data']
        ids = sorted({m['id'] for m in data if isinstance(m, dict) and isinstance(m.get('id'), str)})
        if not ids or any(not re.fullmatch(r'[A-Za-z0-9_.:/-]+', m) for m in ids):
            raise ValueError('Invalid model list.')
        return ids
    except HTTPError as error:
        detail = error.read(4096).decode('utf-8', errors='replace').replace(key, '[REDACTED]')
        raise ValueError(f'Account model lookup failed (HTTP {error.code}): {detail}') from None
    except (URLError, OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Cannot read this account\'s /models list ({type(error).__name__}).') from None


def resolve_model(name, ids):
    for candidate in PRESETS[name]['ids']:
        matches = [m for m in ids if m.lower().rsplit('/', 1)[-1] == candidate]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError('Multiple routes serve this model. Choose --model-id: ' + ', '.join(matches))
    raise ValueError(f'{name} was not found in this account. Use --model-id only for the exact model you intend to test.')


def defaults(name, video=None):
    preset = PRESETS[name]
    delivery = preset['delivery']
    if video == 'native':
        if not preset['native']:
            raise ValueError(f'{name} does not accept native video. Use its default 180-image route.')
        delivery = 'native'
    elif video == 'images':
        delivery = 'ordered-images'
    return dict(video_delivery=delivery, token_field=preset['token_field'], image_detail=preset['detail'],
                reasoning_effort=preset['reasoning'], api_format=preset.get('api_format', 'chat'),
                enable_thinking=preset.get('enable_thinking'),
                max_tokens=4096, concurrency=1)


def messages_body(body):
    """Native Anthropic input for a gateway exposing /v1/messages."""
    content = []
    for part in body['messages'][0]['content']:
        if part['type'] == 'text':
            content.append(part)
        elif part['type'] == 'image_url':
            url = part['image_url']['url']
            if trusted_download_url(url):
                source = dict(type='url', url=url)
            else:
                header, data = url.split(',', 1)
                if header != 'data:image/png;base64': raise ValueError('Expected lossless PNG input.')
                source = dict(type='base64', media_type='image/png', data=data)
            content.append(dict(type='image', source=source,
                                transformations={'oversized_image': 'error'}))
        else:
            raise ValueError('Native Messages requires ordered images, not a video file.')
    result = dict(model=body['model'], messages=[dict(role='user', content=content)],
                  max_tokens=body.get('max_tokens', body.get('max_completion_tokens')), stream=False)
    if 'output_config' in body:
        result['output_config'] = body['output_config']
    return result
