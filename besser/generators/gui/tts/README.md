# Design-time GUI speech

This follows the agent framework's `Text2Speech` abstraction with separate
`OpenAIText2Speech` and `PiperText2Speech` adapters. It does not instantiate an
agent or import optional agent NLP dependencies. Providers return WAV bytes
through the common `SpeechAudio` contract. Add another adapter and extend the
environment factory to support another synthesis engine.

The platform operator chooses the provider and model in the backend environment.
The browser cannot override the provider, service URL, voice, or credentials.
Generation is an explicit action and never happens while typing or personalizing.

At startup, a missing `BESSER_GUI_TTS_PROVIDER` triggers the standard optional
environment warning that related functionality is unavailable. Unset settings
with defaults are logged at INFO with the default being used. Piper additionally
warns when `BESSER_GUI_TTS_LANGUAGE` is absent, since explicit language requests
then cannot be matched to the deployed voice. Invalid provider configurations
produce a warning without stopping unrelated features or logging secrets.

## OpenAI

Set these in the same terminal that launches the backend:

```powershell
$env:BESSER_GUI_TTS_PROVIDER = "openai"
$env:BESSER_GUI_TTS_MODEL = "gpt-4o-mini-tts"
$env:BESSER_GUI_TTS_VOICE = "coral"
$env:BESSER_GUI_TTS_SPEED = "1"
# OPENAI_API_KEY must already be configured in this environment.
```

Models implemented: `gpt-4o-mini-tts`, `tts-1`, and `tts-1-hd`.
The default model for the OpenAI provider is `gpt-4o-mini-tts`; its default voice
is `coral`. Legacy models default to `alloy`. Speed accepts 0.25 to 4.0.
The SDK is imported lazily. Requests follow the official speech API:
https://developers.openai.com/api/docs/guides/text-to-speech

## Piper service

This adapter supports the PCM16 HTTP service used by the agent framework's
Piper provider. The deployed model must match the declared settings:

```powershell
$env:BESSER_GUI_TTS_PROVIDER = "piper"
$env:BESSER_GUI_TTS_MODEL = "mbarnig/lb_rhasspy_piper_tts"
$env:BESSER_GUI_TTS_URL = "http://localhost:8000/synthesize"
$env:BESSER_GUI_TTS_SAMPLE_RATE = "22500"
$env:BESSER_GUI_TTS_LANGUAGE = "lb"
$env:BESSER_GUI_TTS_SPEED = "1"
```

The service receives `POST {"text": "..."}` and returns WAV or mono
little-endian PCM16 bytes (`audio/l16`), as in the framework adapter. The sample
rate must match the service. Piper's deployed voice fixes the language;
incompatible explicit language requests fail instead of silently using the
wrong voice. GUI language names and matching language codes are accepted.
The service has no speed control; non-default speed is rejected.

## Designer workflow and assets

1. Enable GUI > Modality > Read text aloud in the user profile to add listening controls.
2. Apply that profile to the GUI page with Apply defined GUI personalization rules
   selected (it is selected by default when rules exist). Enabled text blocks and
   summaries show inline Play/Pause and Stop controls below their text. Clicking Play requests that block's
   preview and starts playback, or asks you to press Play if browser policy blocks it.
3. If Read text aloud is enabled, web app export synthesizes audio from the final
   modeled text into the generated app's `public/audio` directory.

Preview controls appear only on active personalized variants with an applied
Read text aloud request. Base pages and variants without that request hide
the canvas controls. Generation progress and errors appear beside the inline buttons.
Text
blocks and summaries are separate listening targets. Labels, headings, forms,
table data, code, templates and bound content are excluded. Text is synthesized
after personalization; no LLM rewrites it in the speech step. Long blocks are
split below the OpenAI input limit and WAV chunks are combined into one clip.
The exported React component provides matching Play/Pause/Stop controls, no autoplay, keyboard
accessible native buttons, an AI-voice disclosure, and only one active reader.
Canvas playback controls are DOM-only and never enter the saved GrapesJS model or
HTML export. They are refreshed when text, variants, or the canvas frame change.

Each preview request synthesizes audio on demand and returns the clips directly
as base64 WAV strings. The response uses `Cache-Control: no-store`. The backend
does not write preview files, retain a cache, or add audio references to the model.
The editor creates temporary Blob URLs for playback and reuses them for Play/Pause
and Stop while the same page and variant remain active. Switching pages, variants,
modeling perspectives, or browser tabs, or closing the editor releases playback and
revokes those URLs. Editing a text block also invalidates its preview. Pending requests
are aborted and late responses discarded. Failed requests leave the model unchanged.
`BESSER_GUI_AUDIO_CACHE_DIR` and the old asset-serving endpoint are no longer used.

Preview is independent of export. Enabling the Read text aloud rule means export
will call the configured TTS provider for the latest modeled text; no preview is
required first. Only the generated application output contains persistent audio
files. Identical blocks within an export may share a generated asset. Exported
apps play their own assets with no TTS API or modeling-server dependency.
Runtime-generated/bound text is excluded. Old preview cache files are no longer
used; this change does not delete existing files on the operator's machine.
