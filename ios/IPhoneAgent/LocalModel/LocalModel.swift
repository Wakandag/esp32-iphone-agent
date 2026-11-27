import Foundation

/// The on-device model: llama.cpp over a GGUF bundled on the device. Chosen by the model id `local`,
/// so nothing else in the app changes. With a projector (mmproj.gguf) it also sees screenshots.
enum LocalModelSupport {
    static let modelID = "local"

    static var isAvailable: Bool {
        #if canImport(llama)
        return LocalLLM.modelURL != nil
        #else
        return false
        #endif
    }

    static var modelName: String {
        Self.isAvailable ? "Qwen2-VL abliterated (on-device)" : "no on-device model"
    }

    static func chat(system: String, text: String, image: Data?, maxTokens: Int,
                     temperature: Double) async throws -> ChatResult {
        #if canImport(llama)
        let runner = LocalLLM.shared
        return try await Task.detached(priority: .userInitiated) {
            try runner.generate(system: system, text: text, image: image,
                                maxTokens: maxTokens, temperature: temperature)
        }.value
        #else
        throw LLMError(message: "This build has no on-device model bundled.")
        #endif
    }
}

#if canImport(llama)
import llama
import mtmd

/// One GGUF text model (plus an optional vision projector), loaded once and used for one turn per call.
/// CPU only: no Metal, which keeps the iOS build simple.
final class LocalLLM: @unchecked Sendable {
    static let shared = LocalLLM()

    private let lock = NSLock()
    private var model: OpaquePointer?
    private var projector: OpaquePointer?
    private var projectorTried = false

    // MARK: files

    /// `Documents/model.gguf` first (pushed over the filesystem, survives reinstalls), then the bundle.
    static var modelURL: URL? {
        find("model.gguf")
    }

    /// `Documents/mmproj.gguf`: without it the model answers text only.
    static var projectorURL: URL? {
        find("mmproj.gguf")
    }

    private static func find(_ name: String) -> URL? {
        let files = FileManager.default
        if let docs = files.urls(for: .documentDirectory, in: .userDomainMask).first {
            let pushed = docs.appendingPathComponent(name)
            if files.fileExists(atPath: pushed.path) { return pushed }
        }
        if let support = files.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let local = support.appendingPathComponent(name)
            if files.fileExists(atPath: local.path) { return local }
        }
        let base = (name as NSString).deletingPathExtension
        return Bundle.main.url(forResource: base, withExtension: "gguf")
            ?? Bundle.main.url(forResource: base, withExtension: "gguf", subdirectory: "Resources")
    }

    // MARK: loading

    private func loadedModel() throws -> OpaquePointer {
        if let model { return model }
        guard let url = Self.modelURL else {
            throw LLMError(message: "model.gguf is missing from this device.")
        }
        var params = llama_model_default_params()
        params.n_gpu_layers = 0
        guard let loaded = llama_model_load_from_file(url.path, params) else {
            throw LLMError(message: "The on-device model failed to load (out of memory?).")
        }
        model = loaded
        return loaded
    }

    /// The vision projector, or nil when this device has none (text-only replies then).
    private func loadedProjector(_ model: OpaquePointer) -> OpaquePointer? {
        if let projector { return projector }
        if projectorTried { return nil }
        projectorTried = true
        guard let url = Self.projectorURL else { return nil }
        var params = mtmd_context_params_default()
        params.use_gpu = false
        params.print_timings = false
        params.n_threads = 4
        guard let ctx = mtmd_init_from_file(url.path, model, params), mtmd_support_vision(ctx) else {
            return nil
        }
        projector = ctx
        return ctx
    }

    // MARK: one turn

    func generate(system: String, text: String, image: Data?, maxTokens: Int,
                  temperature: Double) throws -> ChatResult {
        lock.lock()
        defer { lock.unlock() }

        let started = Date()
        let model = try loadedModel()
        let vocab = llama_model_get_vocab(model)

        var cparams = llama_context_default_params()
        cparams.n_ctx = 4096
        cparams.n_batch = 512
        cparams.n_threads = 4
        cparams.n_threads_batch = 4
        guard let ctx = llama_init_from_model(model, cparams) else {
            throw LLMError(message: "The on-device model could not start.")
        }
        defer { llama_free(ctx) }

        var promptTokens = 0
        var position: llama_pos = 0
        let vision = image.flatMap { data -> (OpaquePointer, Data)? in
            guard let projector = loadedProjector(model) else { return nil }
            return (projector, data)
        }

        if let (projector, jpeg) = vision {
            (promptTokens, position) = try prefillWithImage(projector: projector, ctx: ctx,
                                                            system: system, text: text, image: jpeg)
        } else {
            (promptTokens, position) = try prefillText(ctx: ctx, vocab: vocab, system: system, text: text,
                                                       maxTokens: maxTokens)
        }

        let prefilled = Date()
        let nVocab = Int(llama_vocab_n_tokens(vocab))
        var output = ""
        var generated = 0
        var token: llama_token = -1
        // Explicit positions: an image turn uses M-RoPE, so the text after it cannot rely on
        // auto-numbered positions.
        let batch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(batch) }

        while generated < maxTokens {
            if token >= 0 {
                batch.n_tokens = 1
                batch.token[0] = token
                batch.pos[0] = position
                batch.n_seq_id[0] = 1
                batch.seq_id[0]![0] = 0
                batch.logits[0] = 1
                if llama_decode(ctx, batch) != 0 { break }
                position += 1
            }
            guard let logits = llama_get_logits_ith(ctx, -1) else { break }
            let buffer = UnsafeBufferPointer(start: logits, count: nVocab)
            token = Self.sample(buffer, temperature: temperature)
            var piece = [CChar](repeating: 0, count: 128)
            let written = llama_token_to_piece(vocab, token, &piece, 127, 0, true)
            if written <= 0 { break }
            output += String(decoding: piece[0..<Int(written)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
            generated += 1
            if output.contains("<|im_end|>") { break }
        }

        if let cut = output.range(of: "<|im_end|>") { output = String(output[..<cut.lowerBound]) }
        output = output.replacingOccurrences(of: "<|im_start|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        DebugLog.write("local model: \(generated) tokens out in \(String(format: "%.1f", Date().timeIntervalSince(started))) s: \(output.prefix(120))")
        return ChatResult(text: output,
                          firstToken: prefilled.timeIntervalSince(started),
                          total: Date().timeIntervalSince(started),
                          finishReason: generated >= maxTokens ? "length" : "stop",
                          usage: ["prompt_tokens": promptTokens, "completion_tokens": generated])
    }

    /// Text-only turn: ChatML prompt, decoded in batches.
    private func prefillText(ctx: OpaquePointer, vocab: OpaquePointer?, system: String, text: String,
                             maxTokens: Int) throws -> (tokens: Int, position: llama_pos) {
        let prompt = "<|im_start|>system\n\(system)<|im_end|>\n"
            + "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        var tokens = Self.tokenize(vocab, prompt)
        let budget = 4096 - max(64, maxTokens)
        if tokens.count > budget { tokens = Array(tokens.suffix(budget)) }
        guard !tokens.isEmpty else { throw LLMError(message: "The prompt was empty.") }

        var index = 0
        while index < tokens.count {
            let take = min(256, tokens.count - index)
            var chunk = Array(tokens[index..<(index + take)])
            var batch = llama_batch_get_one(&chunk, Int32(take))
            if llama_decode(ctx, batch) != 0 { throw LLMError(message: "On-device decode failed.") }
            index += take
        }
        DebugLog.write("local model: text prompt \(tokens.count) tokens")
        return (tokens.count, llama_pos(tokens.count))
    }

    /// One screenshot + text: llama.cpp's multimodal path (image encoder chunks, then text).
    private func prefillWithImage(projector: OpaquePointer, ctx: OpaquePointer,
                                  system: String, text: String, image: Data) throws -> (tokens: Int, position: llama_pos) {
        let marker = String(cString: mtmd_get_marker(projector))
        let prompt = "<|im_start|>system\n\(system)<|im_end|>\n"
            + "<|im_start|>user\n\(marker)\n\(text)<|im_end|>\n<|im_start|>assistant\n"

        let options = mtmd_helper_init_opt_default()
        let wrapper = image.withUnsafeBytes { buffer -> mtmd_helper_bitmap_wrapper in
            let base = buffer.bindMemory(to: UInt8.self).baseAddress
            return mtmd_helper_bitmap_init_from_buf(projector, base, image.count, false, options)
        }
        guard let bitmap = wrapper.bitmap else {
            throw LLMError(message: "The screenshot could not be read by the on-device model.")
        }
        defer { mtmd_bitmap_free(bitmap) }

        guard let chunks = mtmd_input_chunks_init() else {
            throw LLMError(message: "The on-device model could not prepare the prompt.")
        }
        defer { mtmd_input_chunks_free(chunks) }

        var promptText = Array(prompt.utf8CString)
        var input = mtmd_input_text(text: nil, text_len: 0, add_special: true, parse_special: true)
        let prepared = promptText.withUnsafeMutableBufferPointer { buffer -> Int32 in
            input.text = buffer.baseAddress.map { UnsafePointer($0) }
            input.text_len = prompt.utf8.count
            var bitmaps: [OpaquePointer?] = [bitmap]
            return bitmaps.withUnsafeMutableBufferPointer { pointer in
                mtmd_tokenize(projector, chunks, &input, pointer.baseAddress, 1)
            }
        }
        guard prepared == 0 else {
            throw LLMError(message: "The on-device model could not tokenize the screenshot (code \(prepared)).")
        }

        var past: llama_pos = 0
        let evaluated = mtmd_helper_eval_chunks(projector, ctx, chunks, 0, 0, 512, true, &past)
        guard evaluated == 0 else {
            throw LLMError(message: "The on-device model could not read the screenshot (code \(evaluated)).")
        }
        let tokens = Int(mtmd_helper_get_n_tokens(chunks))
        DebugLog.write("local model: image prompt \(tokens) tokens, \(mtmd_input_chunks_size(chunks)) chunks")
        return (tokens, mtmd_helper_get_n_pos(chunks))
    }

    // MARK: llama.cpp helpers

    private static func tokenize(_ vocab: OpaquePointer?, _ string: String) -> [llama_token] {
        var chars = Array(string.utf8CString)
        let length = Int32(chars.count - 1)
        guard length > 0 else { return [] }
        return chars.withUnsafeBufferPointer { buffer -> [llama_token] in
            guard let base = buffer.baseAddress else { return [] }
            let needed = llama_tokenize(vocab, base, length, nil, 0, true, true)
            let count = needed < 0 ? Int(-needed) : Int(needed)
            guard count > 0 else { return [] }
            var tokens = [llama_token](repeating: 0, count: count)
            let got = llama_tokenize(vocab, base, length, &tokens, Int32(count), true, true)
            guard got > 0 else { return [] }
            return Array(tokens.prefix(Int(got)))
        }
    }

    /// Greedy at temperature 0, else top-k 40 → top-p 0.9 → draw.
    private static func sample(_ logits: UnsafeBufferPointer<Float>, temperature: Double) -> llama_token {
        let n = logits.count
        let k = min(40, n)

        if temperature <= 0.01 {
            var best = 0
            var bestValue = -Float.infinity
            for i in 0..<n where logits[i] > bestValue { bestValue = logits[i]; best = i }
            return llama_token(best)
        }

        var bestIndex = [Int](repeating: 0, count: k)
        var bestValue = [Float](repeating: -Float.infinity, count: k)
        var filled = 0
        for i in 0..<n {
            let value = logits[i]
            if filled < k {
                var slot = filled
                while slot > 0 && bestValue[slot - 1] < value {
                    bestValue[slot] = bestValue[slot - 1]
                    bestIndex[slot] = bestIndex[slot - 1]
                    slot -= 1
                }
                bestValue[slot] = value
                bestIndex[slot] = i
                filled += 1
            } else if value > bestValue[k - 1] {
                var slot = k - 1
                while slot > 0 && bestValue[slot - 1] < value {
                    bestValue[slot] = bestValue[slot - 1]
                    bestIndex[slot] = bestIndex[slot - 1]
                    slot -= 1
                }
                bestValue[slot] = value
                bestIndex[slot] = i
            }
        }

        let top = min(filled, k)
        guard top > 0 else { return 0 }
        let peak = bestValue[0]
        var weights = (0..<top).map { exp(Double(bestValue[$0] - peak) / temperature) }
        let sum = weights.reduce(0, +)
        guard sum > 0 else { return llama_token(bestIndex[0]) }
        weights = weights.map { $0 / sum }

        var cumulative = 0.0
        var keep = top
        for (i, w) in weights.enumerated() {
            cumulative += w
            if cumulative > 0.9 { keep = i + 1; break }
        }
        let keptTotal = weights.prefix(keep).reduce(0, +)
        var draw = Double.random(in: 0..<keptTotal)
        for i in 0..<keep {
            draw -= weights[i]
            if draw <= 0 { return llama_token(bestIndex[i]) }
        }
        return llama_token(bestIndex[0])
    }
}
#endif
