// electron/audio/whisper/__tests__/EnglishOnlyModelLanguageOption.test.mjs
//
// THE BUG THIS PINS: an English-only Whisper checkpoint was sent
// `language: 'english'`, and transformers.js refuses BOTH `language` and `task`
// on those models:
//
//   Error: Cannot specify `task` or `language` for an English-only model.
//     at closure._retrieve_init_tokens (transformers.node.mjs:10971)
//
// It throws in _retrieve_init_tokens, before any audio is decoded, so the
// failure is total: every segment fails and the channel emits no text at all
// while capture, VAD and dispatch all look perfectly healthy. From a packaged
// Windows build — three dispatches, three failures, zero words:
//
//   transcribe START  task=t1 lang=english model=Xenova/whisper-tiny.en
//   transcribe FAILED task=t1: Cannot specify `task` or `language` ...
//   transcribe START  task=t2 lang=english model=Xenova/whisper-tiny.en
//   transcribe FAILED task=t2: Cannot specify `task` or `language` ...
//
// The forcing was well-intentioned — it stopped a multilingual selection from
// asking an English-only decoder for Russian — but 'english' is not a legal
// value to pass to these models either. The correct request omits both options;
// they transcribe English unconditionally, so there is nothing to select.
//
// Run: node --test 'electron/audio/whisper/__tests__/EnglishOnlyModelLanguageOption.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(__dirname, '../whisperWorker.ts');
const source = fs.readFileSync(SRC, 'utf8');

describe('English-only Whisper checkpoints', () => {
  test('never forces language to "english" — that is the value that throws', () => {
    // The exact shape of the old bug: `language = 'english'` inside the
    // English-only branch. Anything that reintroduces it fails here.
    assert.ok(
      !/language\s*=\s*['"]english['"]/.test(source),
      'English-only models must be sent NO language at all, not language="english" ' +
      '— transformers.js throws "Cannot specify `task` or `language` for an English-only model"',
    );
  });

  test('clears the language for English-only checkpoints', () => {
    assert.match(
      source,
      /const\s+englishOnly\s*=\s*isEnglishOnlyModel\(loadedModelId\)[\s\S]{0,200}?if\s*\(englishOnly\)\s*language\s*=\s*null/,
      'the English-only branch must null the language out',
    );
  });

  test('drops `task` too — it is refused alongside `language`', () => {
    assert.match(
      source,
      /if\s*\(englishOnly\)\s*delete\s+opts\.task/,
      'opts.task is set unconditionally for every request, so English-only ' +
      'models need it removed or they throw on task alone',
    );
  });

  test('detects English-only from the model config, not just a hardcoded list', () => {
    // A hardcoded id list cannot know about a checkpoint the user adds later,
    // and being wrong costs the entire session's transcript.
    assert.match(
      source,
      /generation_config\?\.\s*is_multilingual\s*===\s*false/,
      'must consult the same flag transformers.js itself checks',
    );
  });

  test('a refusal is recorded and the segment retried, so only one is lost', () => {
    assert.match(source, /englishOnlyAtRuntime\.add\(loadedModelId\)/);
    assert.match(
      source,
      /isEnglishOnlyRefusal\(e\)[\s\S]{0,400}?result\s*=\s*await\s+pipe\(msg\.audio,\s*opts\)/,
      'an unlisted English-only model must cost one segment, not the session',
    );
  });

  test('the static list still covers the .en and Distil checkpoints', () => {
    for (const id of [
      'Xenova/whisper-tiny.en',
      'Xenova/whisper-base.en',
      'Xenova/whisper-small.en',
      'Xenova/whisper-medium.en',
      'distil-whisper/distil-small.en',
      'onnx-community/moonshine-tiny-ONNX',
    ]) {
      assert.ok(source.includes(`'${id}'`), `${id} must stay in ENGLISH_ONLY_MODELS`);
    }
  });

  test('multilingual models keep their language — the Russian fix is not undone', () => {
    // resolveWhisperLanguage() exists precisely so a Russian selection reaches
    // the decoder. The English-only guard must not swallow it for every model.
    assert.match(source, /let\s+language:\s*string\s*\|\s*null\s*=\s*resolveWhisperLanguage\(msg\.language\)/);
    assert.match(source, /if\s*\(language\)\s*opts\.language\s*=\s*language/);
  });
});
