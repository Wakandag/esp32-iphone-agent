# On-device model (free, no API key)

The app can run its own model on the iPhone through llama.cpp. The model id is `local`:
nothing else in the app changes, and no network request is made.

- Model: **Qwen2.5-1.5B-Instruct abliterated**, `Q4_K_M` (986 MB), Apache-2.0
  (<https://huggingface.co/Vali/Qwen2.5-1.5B-Instruct-abliterated-Q4_K_M-GGUF>)
- Runtime: llama.cpp `v1.0.0`, arm64, CPU only — `scripts/build-llama.sh`
- Text only: screenshots are not sent to it, the prompt's text is.

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
reinstall). The app reads `Documents/model.gguf` first, then `Application Support/model.gguf`,
then the bundle:

```bash
./scripts/fetch-model.sh                                  # downloads model.gguf locally
python3 scripts/push-model.py model.gguf --host <phone>   # chunked upload into the app container
```

## Point the app at it

Settings → Model → Provider **Custom Endpoint**, Base URL `https://local.invalid/v1`
(any valid https URL; it is never contacted), Model `local`.

The build seeds these through `IA_PRESET_MODEL`/`IA_PRESET_BASE_URL` in
`Config/Shared.xcconfig` for a fresh install (stored settings win once they exist).

`Test Connection` runs one real completion; on this phone it answered in about 2.4 s and
reports "the image check was inconclusive" because the model is text-only.
