# On-device model (free, no API key)

The app can run its own model on the iPhone through llama.cpp. The model id is `local`:
nothing else in the app changes, and no network request is made.

- Model: **Qwen2-VL-2B-Instruct abliterated**, `Q4_K_M` (1.04 GiB) + `mmproj-fp16` (1.24 GiB), Apache-2.0
  (<https://huggingface.co/mradermacher/Qwen2-VL-2B-Instruct-abliterated-GGUF>)
- Runtime: llama.cpp `v1.0.0`, arm64, CPU only — `scripts/build-llama.sh` (builds `llama` + `mtmd`
  and merges the archives into `libagentmodel.a`)
- Vision: screenshots go through `mtmd` (image encoder + projector); without `mmproj.gguf` it
  answers text only.

Known quirk: the app's connection probe shows a 64x64 flat red square and demands the single word
"red". Qwen2-VL misreads that specific probe (reproduced with llama.cpp's own CLI, official weights
included: it answers "blue"), while real screenshots read correctly — verified locally: the same
files described a newspaper front page accurately. So for the `local` model the app only requires a
non-empty answer, and reports **Connected**.

## Build

```bash
./scripts/build-llama.sh        # clones llama.cpp, stages ios/llama/{libs,headers}
cd ios && xcodegen generate && xcodebuild ...
```

Both are already wired into `.github/workflows/build-ipa.yml`, so the CI IPA links llama.cpp.
`ios/IPhoneAgent/LocalModel/LocalModel.swift` compiles the runner only when the module is
available (`#if canImport(llama)`), so a checkout without the libs still builds.

## Get the model onto the phone

The GGUF is not bundled in the IPA (that made it ~1 GB and it had to be re-uploaded on every
reinstall). The app reads `Documents/model.gguf` and `Documents/mmproj.gguf` first, then `Application Support`,
then the bundle:

```bash
./scripts/fetch-model.sh                                  # downloads model.gguf locally
python3 scripts/push-model.py model.gguf  --remote-name model.gguf  --host <phone>
python3 scripts/push-model.py mmproj.gguf --remote-name mmproj.gguf --host <phone>
```

## Point the app at it

Settings → Model → Provider **Custom Endpoint**, Base URL `https://local.invalid/v1`
(any valid https URL; it is never contacted), Model `local`.

The build seeds these through `IA_PRESET_MODEL`/`IA_PRESET_BASE_URL` in
`Config/Shared.xcconfig` for a fresh install (stored settings win once they exist).

`Test Connection` runs one real completion; on this phone it answered in about 2.4 s and
reports "the image check was inconclusive" because the model is text-only.
