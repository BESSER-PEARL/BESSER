import asyncio
import base64
import io
import json
import wave
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from besser.generators.gui.gui_audio import generate_gui_audio
from besser.generators.gui.tts.base import SpeechAudio, Text2Speech
from besser.generators.gui.tts import create_text2speech
from besser.generators.gui.tts.openai import OpenAIText2Speech
from besser.generators.gui.tts.piper import PiperText2Speech


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b'\x00\x00' * 100)
    return buffer.getvalue()


class FakeTTS(Text2Speech):
    def __init__(self, voice='test'):
        self.voice = voice
        self.calls = []

    @property
    def configuration(self):
        return {'provider': 'fake', 'model': 'test', 'voice': self.voice, 'speed': 1}

    def text2speech(self, text, language=None):
        self.calls.append((text, language))
        return SpeechAudio(wav_bytes())


def page(text='Read these instructions before you start.'):
    return {'components': [{'type': 'text', 'tagName': 'p', 'content': text, 'attributes': {'id': 'instructions'}}], 'css': []}


def test_generates_grouped_prose_and_summary_without_modifying_original():
    snapshot = page('<strong>Read these instructions</strong> before you start.')
    snapshot['components'].append({'type': 'text', 'content': 'A short summary of the steps.', 'attributes': {'data-besser-gui-summary': 'true'}})
    original = deepcopy(snapshot)
    provider = FakeTTS()
    result = generate_gui_audio(snapshot, provider, language='english')
    assert result['blocks'] == len(result['clips']) == 2
    assert provider.calls == [('Read these instructions before you start.', 'english'), ('A short summary of the steps.', 'english')]
    assert base64.b64decode(result['clips'][0]['audio']) == wav_bytes()
    assert result['clips'][1]['isSummary'] is True
    assert snapshot == original
    assert 'guiPage' not in result


def test_each_preview_is_generated_on_demand_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = FakeTTS()
    generate_gui_audio(page(), provider)
    generate_gui_audio(page(), provider)
    assert len(provider.calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_skips_controls_dynamic_content_forms_and_table_data():
    snapshot = {'components': [
        {'type': 'button', 'content': 'Click to start this task'},
        {'type': 'table', 'data': [{'text': 'Do not read this row'}]},
        {'type': 'text', 'content': 'Hello {{user.name}}, welcome back'},
        {'type': 'text', 'content': 'Your profile name appears here', 'attributes': {'data-binding': 'user.name'}},
        {'tagName': 'form', 'components': page()['components']},
        {'type': 'text', 'tagName': 'h1', 'content': 'A long heading is not a block'},
    ], 'css': []}
    provider = FakeTTS()
    assert generate_gui_audio(snapshot, provider)['blocks'] == 0
    assert provider.calls == []


def test_long_text_is_chunked_and_wav_chunks_are_combined():
    provider = FakeTTS()
    result = generate_gui_audio(page('Read this sentence carefully. ' * 400), provider)
    assert len(provider.calls) > 1
    assert all(len(text) <= 3900 for text, _ in provider.calls)
    with wave.open(io.BytesIO(base64.b64decode(result['clips'][0]['audio'])), 'rb') as stream:
        assert stream.getnframes() == 100 * len(provider.calls)


def test_provider_failure_does_not_change_the_input():
    snapshot = page()
    original = deepcopy(snapshot)
    with patch.object(FakeTTS, 'text2speech', side_effect=RuntimeError('provider unavailable')):
        with pytest.raises(RuntimeError): generate_gui_audio(snapshot, FakeTTS())
    assert snapshot == original


def test_provider_selection_is_server_controlled(monkeypatch):
    monkeypatch.delenv('BESSER_GUI_TTS_PROVIDER', raising=False)
    with pytest.raises(ValueError, match='BESSER_GUI_TTS_PROVIDER'): create_text2speech()
    monkeypatch.setenv('BESSER_GUI_TTS_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.delenv('BESSER_GUI_TTS_MODEL', raising=False)
    provider = create_text2speech()
    assert provider.configuration['model'] == 'gpt-4o-mini-tts'
    assert 'test-key' not in json.dumps(provider.configuration)
    monkeypatch.setenv('BESSER_GUI_TTS_MODEL', 'tts-1-hd')
    assert create_text2speech().configuration['model'] == 'tts-1-hd'
    monkeypatch.setenv('BESSER_GUI_TTS_PROVIDER', 'piper')
    monkeypatch.setenv('BESSER_GUI_TTS_MODEL', 'deployed-piper-model')
    assert isinstance(create_text2speech(), PiperText2Speech)


def test_openai_adapter_requests_mini_tts_audio_and_target_language():
    with patch('openai.OpenAI') as client:
        speech = client.return_value.__enter__.return_value.audio.speech.create
        speech.return_value.content = wav_bytes()
        assert OpenAIText2Speech(api_key='test-key').text2speech('Bonjour tout le monde.', 'french').content == wav_bytes()
        assert speech.call_args.kwargs['model'] == 'gpt-4o-mini-tts'
        assert 'french' in speech.call_args.kwargs['instructions']


def test_piper_adapter_wraps_framework_pcm_as_wav():
    response = MagicMock(content=b'\x00\x00' * 100, headers={'content-type': 'audio/l16'})
    provider = PiperText2Speech(url='http://localhost/synthesize', model='piper', language='luxembourgish')
    with patch('requests.post', return_value=response) as post:
        audio = provider.text2speech('Moien alleguer.', 'luxembourgish')
    assert post.call_args.kwargs['json'] == {'text': 'Moien alleguer.'}
    with wave.open(io.BytesIO(audio.content), 'rb') as stream:
        assert stream.getframerate() == 22500
    with pytest.raises(ValueError, match='language'):
        provider.text2speech('Bonjour.', 'french')


def test_audio_endpoint_returns_audio_directly_and_disables_http_caching():
    # Match backend startup order; direct router-first import hits an existing deployment cycle.
    import besser.utilities.web_modeling_editor.backend.services
    from besser.utilities.web_modeling_editor.backend.routers.generation_router import generate_gui_audio_endpoint
    with patch('besser.generators.gui.tts.create_text2speech', return_value=FakeTTS()):
        result = asyncio.run(generate_gui_audio_endpoint({'guiPage': page(), 'model': 'browser-cannot-select'}))
    response = json.loads(result.body)
    assert base64.b64decode(response['clips'][0]['audio']) == wav_bytes()
    assert result.headers['cache-control'] == 'no-store'
    assert 'guiPage' not in response
