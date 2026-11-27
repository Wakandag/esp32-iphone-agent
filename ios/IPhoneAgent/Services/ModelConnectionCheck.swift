import Foundation

/// An unsuccessful image check does not imply an authentication or connectivity failure.
enum ModelConnectionCheck {
    /// `onDevice`: the bundled model. The colour probe is built for hosted vision models — Qwen2-VL
    /// misreads a 64x64 flat red square (reproduced with llama.cpp's own CLI, official weights too),
    /// while it reads real screenshots correctly — so a non-empty answer is what counts there.
    static func validate(_ reply: ChatResult, onDevice: Bool = false) throws {
        let answer = reply.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !answer.isEmpty else {
            throw LLMError(message: String(localized: "Connected, but the model returned no answer. Try again or choose another vision model."))
        }
        if onDevice { return }
        let words = answer.lowercased().split { !$0.isLetter }
        guard words == ["red"] else {
            throw LLMError(message: String(localized: "Connected, but the image check was inconclusive. Try again or choose a specific vision model.")
                           + "\n" + String(localized: "Model reply:") + " " + String(answer.prefix(160)))
        }
    }
}
