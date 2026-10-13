import logging
from unittest.mock import patch

from besser.generators.gui.tts import log_text2speech_environment


def test_openai_default_settings_are_available_and_do_not_warn_or_contact_provider(caplog):
    with patch.dict('os.environ', {'BESSER_GUI_TTS_PROVIDER': 'openai', 'OPENAI_API_KEY': 'secret-test-key'}, clear=True), caplog.at_level(logging.INFO):
        log_text2speech_environment(logging.getLogger('tts-environment-test'))
    assert 'Using default gpt-4o-mini-tts' in caplog.text
    assert 'Using default coral' in caplog.text
    assert 'Using default 1' in caplog.text
    assert 'unavailable' not in caplog.text
    assert 'secret-test-key' not in caplog.text


def test_piper_warns_for_missing_language_without_openai_settings(caplog):
    with patch.dict('os.environ', {'BESSER_GUI_TTS_PROVIDER': 'piper'}, clear=True), caplog.at_level(logging.INFO):
        log_text2speech_environment(logging.getLogger('tts-environment-test'))
    assert 'BESSER_GUI_TTS_LANGUAGE is not set' in caplog.text
    assert 'Using default 22500' in caplog.text
    assert 'BESSER_GUI_TTS_VOICE' not in caplog.text


def test_invalid_configuration_warns_without_exposing_environment_values(caplog):
    with patch.dict('os.environ', {'BESSER_GUI_TTS_PROVIDER': 'openai', 'OPENAI_API_KEY': 'secret-test-key', 'BESSER_GUI_TTS_SPEED': 'secret-invalid-value'}, clear=True):
        log_text2speech_environment(logging.getLogger('tts-environment-test'))
    assert 'Related functionality will be unavailable' in caplog.text
    assert 'secret-test-key' not in caplog.text
    assert 'secret-invalid-value' not in caplog.text
