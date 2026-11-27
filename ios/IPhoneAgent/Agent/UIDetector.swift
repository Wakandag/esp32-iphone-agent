import CoreGraphics
import CoreML
import Foundation
import UIKit
import Vision

/// One box on a screenshot. `rect` is normalised to the picture (0–1, origin top left), so the same box fits the
/// screen it came from, any view of it and either orientation: a landscape app's frame is just another rectangle.
struct UIDetection {
    let label: String
    let score: Float
    let rect: CGRect
}

/// Boxes over whatever app the phone shows.
///
/// The detector is a YOLO exported to Core ML and pushed to `Documents/uidetector.mlpackage` (or `.mlmodelc` /
/// `.mlmodel`). The export contract is the one SA-Ai-iOS already ships (its `Shared/SADetector.mm`, ported here):
/// one image input, one multi array of shape (1, C, N) or (1, N, C) holding cx, cy, w, h and a score — in pixels
/// of the model input or in 0–1, whichever the values use. Boxes are filtered by score and merged by IoU.
///
/// Without a model it falls back to what iOS itself reads in the picture — text lines and large rectangles. That
/// needs no weights, works on every app, costs ~50–150 ms, and its boxes are the labels to train the next model on
/// (`scripts/train-ui-detector.py`).
enum UIDetector {
    /// Low scores are noise on a screen: the head detector this contract came from fired on scenery below 0.3.
    static let minConfidence: Float = 0.3
    static let iou = 0.45

    // MARK: model

    private struct Loaded {
        let model: MLModel
        let input: String
        let output: String
        let size: CGSize
    }

    /// `Documents/uidetector.*` first, so a retrained model is a file push and not a new build.
    static var modelURL: URL? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
            let url = docs.appendingPathComponent("uidetector").appendingPathExtension(ext)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    private static let loaded: Loaded? = load()

    private static func load() -> Loaded? {
        guard let url = modelURL else { return nil }
        var compiled = url
        if url.pathExtension != "mlmodelc" {
            guard let made = try? MLModel.compileModel(at: url) else {
                DebugLog.write("detector: \(url.lastPathComponent) did not compile")
                return nil
            }
            compiled = made
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        guard let model = try? MLModel(contentsOf: compiled, configuration: configuration) else {
            DebugLog.write("detector: CoreML refused \(url.lastPathComponent)")
            return nil
        }
        let description = model.modelDescription
        guard let input = description.inputDescriptionsByName.first(where: { $0.value.type == .image }) else {
            DebugLog.write("detector: \(url.lastPathComponent) has no image input")
            return nil
        }
        guard let output = description.outputDescriptionsByName.first(where: { $0.value.type == .multiArray }) else {
            DebugLog.write("detector: \(url.lastPathComponent) has no box output")
            return nil
        }
        let constraint = input.value.imageConstraint
        let size = CGSize(width: constraint?.pixelsWide ?? 640, height: constraint?.pixelsHigh ?? 640)
        DebugLog.write("detector: \(url.lastPathComponent) ready, \(Int(size.width))x\(Int(size.height)), in \(input.key), out \(output.key)")
        return Loaded(model: model, input: input.key, output: output.key, size: size)
    }

    static var isUsingModel: Bool { loaded != nil }

    // MARK: detection

    /// Boxes for a screenshot, in picture space. Empty when nothing was found.
    static func detect(_ jpeg: Data) -> [UIDetection] {
        guard let image = UIImage(data: jpeg)?.cgImage else { return [] }
        if let loaded {
            let boxes = byModel(loaded, image)
            DebugLog.write("detector: \(boxes.count) boxes (trained model)")
            return boxes
        }
        let boxes = byVision(image)
        DebugLog.write("detector: \(boxes.count) boxes (Vision)")
        return boxes
    }

    /// The picture padded to the model's square input: stretching a portrait screen into 640² would squash every
    /// button, and cropping a landscape one would cut the corners off.
    private static func squared(_ image: CGImage, _ side: CGSize) -> (UIImage, CGRect) {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let full = CGRect(origin: .zero, size: side)
        let scale = min(side.width / CGFloat(image.width), side.height / CGFloat(image.height))
        let fitted = CGRect(x: (side.width - CGFloat(image.width) * scale) / 2,
                            y: (side.height - CGFloat(image.height) * scale) / 2,
                            width: CGFloat(image.width) * scale, height: CGFloat(image.height) * scale)
        let padded = UIGraphicsImageRenderer(size: side, format: format).image { context in
            UIColor(white: 0.45, alpha: 1).setFill()
            context.fill(full)
            UIImage(cgImage: image).draw(in: fitted)
        }
        return (padded, fitted)
    }

    private static func byModel(_ loaded: Loaded, _ image: CGImage) -> [UIDetection] {
        let (padded, fitted) = squared(image, loaded.size)
        guard let pixels = padded.cgImage else { return [] }
        guard let vision = try? VNCoreMLModel(for: loaded.model) else { return [] }
        let request = VNCoreMLRequest(model: vision)
        request.imageCropAndScaleOption = .scaleFill  // already square
        try? VNImageRequestHandler(cgImage: pixels, orientation: .up).perform([request])
        let arrays = (request.results ?? []).compactMap {
            ($0 as? VNCoreMLFeatureValueObservation)?.featureValue.multiArrayValue
        }
        guard let array = arrays.first else { return [] }
        let raw = decode(array, size: loaded.size)
        let merged = nonMaximum(raw)
        // Square → picture: undo the padding the model never saw.
        let box = CGSize(width: fitted.width / loaded.size.width, height: fitted.height / loaded.size.height)
        let origin = CGPoint(x: fitted.minX / loaded.size.width, y: fitted.minY / loaded.size.height)
        return merged.compactMap { detection in
            guard box.width > 0, box.height > 0 else { return nil }
            let rect = CGRect(x: (detection.rect.minX - origin.x) / box.width,
                              y: (detection.rect.minY - origin.y) / box.height,
                              width: detection.rect.width / box.width,
                              height: detection.rect.height / box.height)
            let clipped = rect.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard clipped.width > 0.004, clipped.height > 0.004 else { return nil }
            return UIDetection(label: detection.label, score: detection.score, rect: clipped)
        }
    }

    /// Reads the raw YOLO head: (1, C, N) or (1, N, C), pixels or 0–1, score already through a sigmoid or not.
    private static func decode(_ array: MLMultiArray, size: CGSize) -> [UIDetection] {
        let shape = array.shape.map(\.intValue)
        guard shape.count >= 2 else { return [] }
        let a = shape[shape.count - 2], b = shape[shape.count - 1]
        let channelFirst = b > a
        let channels = channelFirst ? a : b
        let anchors = channelFirst ? b : a
        guard channels >= 5, anchors > 0, anchors < 200_000 else { return [] }
        let lead = shape.count > 2 ? Array(shape[0..<(shape.count - 2)]) : []

        func value(_ channel: Int, _ anchor: Int) -> Float {
            let indices = (lead + (channelFirst ? [channel, anchor] : [anchor, channel])).map { NSNumber(value: $0) }
            return array[indices]?.floatValue ?? 0
        }

        var found: [UIDetection] = []
        for anchor in 0..<anchors {
            var score = value(4, anchor)
            if score < 0 || score > 1 { score = 1 / (1 + exp(-score)) }  // older exports skip the sigmoid
            guard score >= minConfidence else { continue }
            let cx = value(0, anchor), cy = value(1, anchor), w = value(2, anchor), h = value(3, anchor)
            guard w > 0, h > 0 else { continue }
            var rect = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2), width: CGFloat(w), height: CGFloat(h))
            if !(cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5) {  // pixels of the model input, not 0–1
                rect = CGRect(x: rect.minX / size.width, y: rect.minY / size.height,
                              width: rect.width / size.width, height: rect.height / size.height)
            }
            guard rect.width > 0.002, rect.height > 0.002 else { continue }
            found.append(UIDetection(label: "ui", score: score, rect: rect))
        }
        return found
    }

    private static func nonMaximum(_ boxes: [UIDetection]) -> [UIDetection] {
        var kept: [UIDetection] = []
        for box in boxes.sorted(by: { $0.score > $1.score }) {
            let overlaps = kept.contains { other in
                let joint = other.rect.intersection(box.rect)
                guard !joint.isNull else { return false }
                let area = joint.width * joint.height
                let total = other.rect.width * other.rect.height + box.rect.width * box.rect.height - area
                return total > 0 && area / total > iou
            }
            if !overlaps { kept.append(box) }
        }
        return Array(kept.prefix(80))
    }

    /// No weights: the text lines and large rectangles iOS itself finds. Works on any app, and is what the next
    /// model is trained on.
    private static func byVision(_ image: CGImage) -> [UIDetection] {
        let text = VNRecognizeTextRequest()
        text.recognitionLevel = .accurate
        text.usesLanguageCorrection = false
        let shapes = VNDetectRectanglesRequest()
        shapes.maximumObservations = 12
        shapes.minimumConfidence = 0.6
        shapes.minimumSize = 0.08
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up)
        try? handler.perform([text, shapes])

        var found: [UIDetection] = (text.results ?? []).compactMap { observation in
            guard let words = observation.topCandidates(1).first?.string, !words.isEmpty else { return nil }
            return UIDetection(label: String(words.prefix(24)), score: observation.confidence, rect: upright(observation.boundingBox))
        }
        found += (shapes.results ?? []).map {
            UIDetection(label: "panel", score: $0.confidence, rect: upright($0.boundingBox))
        }
        return nonMaximum(found)
    }

    /// Vision's boxes start at the bottom left; screenshots here are read from the top left.
    private static func upright(_ box: CGRect) -> CGRect {
        CGRect(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height)
    }

    // MARK: output

    /// The picture with every box drawn on it, for a share sheet or a bug report.
    static func annotated(_ jpeg: Data, boxes: [UIDetection]) -> Data? {
        guard let image = UIImage(data: jpeg) else { return nil }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let size = image.size
        let drawn = UIGraphicsImageRenderer(size: size, format: format).image { context in
            image.draw(in: CGRect(origin: .zero, size: size))
            let cg = context.cgContext
            let width = max(1.5, size.width / 220)
            let font = UIFont.monospacedSystemFont(ofSize: max(9, size.width / 32), weight: .semibold)
            for box in boxes {
                let rect = CGRect(x: box.rect.minX * size.width, y: box.rect.minY * size.height,
                                  width: box.rect.width * size.width, height: box.rect.height * size.height)
                let colour: UIColor = box.score >= 0.6 ? .systemGreen : .systemYellow
                cg.setStrokeColor(colour.cgColor)
                cg.setLineWidth(width)
                cg.stroke(rect.insetBy(dx: -width / 2, dy: -width / 2))
                let label = "\(box.label) \(Int(box.score * 100))%"
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: font, .foregroundColor: UIColor.black, .backgroundColor: colour,
                ]
                let text = label as NSString
                let bounds = text.size(withAttributes: attributes)
                let at = CGPoint(x: rect.minX, y: max(0, rect.minY - bounds.height))
                text.draw(at: at, withAttributes: attributes)
            }
        }
        return drawn.jpegData(compressionQuality: 0.9)
    }

    /// Saves a picture and its boxes as a YOLO sample (`images/`, `labels/`, one line per box), the layout
    /// `scripts/train-ui-detector.py` reads — the same one SA-Ai-iOS trained its detector with.
    @discardableResult
    static func saveSample(_ jpeg: Data, boxes: [UIDetection]) -> URL? {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("uidataset", isDirectory: true)
        let images = root.appendingPathComponent("images", isDirectory: true)
        let labels = root.appendingPathComponent("labels", isDirectory: true)
        try? FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: labels, withIntermediateDirectories: true)
        let name = "shot-\(Int(Date().timeIntervalSince1970))-\(Int.random(in: 100...999))"
        let picture = images.appendingPathComponent(name).appendingPathExtension("jpg")
        let text = labels.appendingPathComponent(name).appendingPathExtension("txt")
        guard (try? jpeg.write(to: picture)) != nil else { return nil }
        let lines = boxes.map { box in
            String(format: "0 %.5f %.5f %.5f %.5f", box.rect.midX, box.rect.midY, box.rect.width, box.rect.height)
        }
        try? (lines.joined(separator: "\n") + "\n").write(to: text, atomically: true, encoding: .utf8)
        return picture
    }
}
