import Foundation

/// The on-device model: llama.cpp over a GGUF bundled in the app. Chosen by the model id `local`,
/// so nothing else in the app changes — Settings just needs model `local` (the build seeds it).
enum LocalModelSupport {
    static let modelID = "local"

    static var isAvailable: Bool {
        #if canImport(llama)
        return LocalLLM.bundleModelURL != nil
        #else
        return false
        #endif
    }

    static var modelName: String {
        Self.isAvailable ? "Qwen2.5 abliterated (on-device)" : "no on-device model"
    }

    /// Text-only: the bundled model sees no screenshots, only the text the agent was going to send.
    static func chat(system: String, text: String, maxTokens: Int, temperature: Double) async throws -> ChatResult {
        #if canImport(llama)
        let runner = LocalLLM.shared
        return try await Task.detached(priority: .userInitiated) {
            try runner.generate(system: system, text: text, maxTokens: maxTokens,
                                temperature: Double(temperature))
        }.value
        #else
        throw LLMError(message: "This build has no on-device model bundled.")
        #endif
    }
}

#if canImport(llama)
import llama

/// Loads the GGUF once and runs one chat turn per call (a fresh context each time, so turns never
/// inherit each other's KV cache). CPU only: no Metal, which keeps the iOS build simple.
final class LocalLLM: @unchecked Sendable {
    static let shared = LocalLLM()

    private let lock = NSLock()
    private var model: OpaquePointer?

    static var bundleModelURL: URL? {
        Bundle.main.url(forResource: "model", withExtension: "gguf")
            ?? Bundle.main.url(forResource: "model", withExtension: "gguf", subdirectory: "Resources")
    }

    private func loadedModel() throws -> OpaquePointer {
        if let model { return model }
        guard let url = Self.bundleModelURL else {
            throw LLMError(message: "model.gguf is missing from this build.")
        }
        var params = llama_model_default_params()
        params.n_gpu_layers = 0
        params.use_mmap = true
        guard let loaded = llama_model_load_from_file(url.path, params) else {
            throw LLMError(message: "The on-device model failed to load (out of memory?).")
        }
        model = loaded
        return loaded
    }

    func generate(system: String, text: String, maxTokens: Int, temperature: Double) throws -> ChatResult {
        lock.lock()
        defer { lock.unlock() }

        let started = Date()
        let model = try loadedModel()
        let vocab = llama_model_get_vocab(model)

        var cparams = llama_context_default_params()
        cparams.n_ctx = 4096
        cparams.n_batch = 256
        cparams.n_threads = 4
        cparams.n_threads_batch = 4
        guard let ctx = llama_init_from_model(model, cparams) else {
            throw LLMError(message: "The on-device model could not start.")
        }
        defer { llama_free(ctx) }

        let prompt = "<|im_start|>system\n\(system)<|im_end|>\n"
            + "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        var tokens = Self.tokenize(vocab, prompt)
        let budget = Int(cparams.n_ctx) - max(64, maxTokens)
        if tokens.count > budget { tokens = Array(tokens.suffix(budget)) }
        guard !tokens.isEmpty else { throw LLMError(message: "The prompt was empty.") }

        // Prefill in batches; a batch's logits land on its last token.
        var index = 0
        var logitsIndex: Int32 = 0
        while index < tokens.count {
            let take = min(256, tokens.count - index)
            var chunk = Array(tokens[index..<(index + take)])
            var batch = llama_batch_get_one(&chunk, Int32(take))
            if llama_decode(ctx, batch) != 0 { throw LLMError(message: "On-device decode failed.") }
            logitsIndex = Int32(take - 1)
            index += take
        }

        let prefilled = Date()
        let nVocab = Int(llama_vocab_n_tokens(vocab))
        var output = ""
        var generated = 0
        var token: llama_token = -1

        while generated < maxTokens {
            if token >= 0 {
                var one: [llama_token] = [token]
                var batch = llama_batch_get_one(&one, 1)
                if llama_decode(ctx, batch) != 0 { break }
                logitsIndex = 0
            }
            guard let logits = llama_get_logits_ith(ctx, logitsIndex) else { break }
            let buffer = UnsafeBufferPointer(start: logits, count: nVocab)
            let next = Self.sample(buffer, temperature: temperature)
            token = next
            var piece = [CChar](repeating: 0, count: 128)
            let written = llama_token_to_piece(vocab, next, &piece, 127, 0, true)
            if written <= 0 { break }
            output += String(decoding: piece[0..<Int(written)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
            generated += 1
            if output.contains("<|im_end|>") { break }
        }

        if let cut = output.range(of: "<|im_end|>") { output = String(output[..<cut.lowerBound]) }
        output = output.replacingOccurrences(of: "<|im_start|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let total = Date().timeIntervalSince(started)
        return ChatResult(text: output,
                          firstToken: prefilled.timeIntervalSince(started),
                          total: total,
                          finishReason: generated >= maxTokens ? "length" : "stop",
                          usage: ["prompt_tokens": tokens.count, "completion_tokens": generated])
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

        // Top-k by insertion: keeps the pass linear over the vocabulary.
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
