import SwiftUI

/// Boxes over the mirrored screen — whichever app it is showing. The frame comes from the same capture the agent
/// looks at, so what is boxed here is exactly what the agent sees; `Detect` runs the trained model when one is
/// pushed to `Documents/uidetector.mlpackage`, else iOS's own text and shape reading.
struct UIDetectorView: View {
    @ObservedObject var feed: VideoFeed
    @Environment(\.dismiss) private var dismiss

    @State private var image: UIImage?
    @State private var source: Data?
    @State private var boxes: [UIDetection] = []
    @State private var busy = false
    @State private var note = "Tap Detect."
    @State private var share: URL?
    @State private var sharing = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                picture
                Text(note).font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 12) {
                    Button { detect() } label: {
                        Label("Detect", systemImage: "viewfinder").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(busy)

                    Button { save() } label: {
                        Label("Save", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(source == nil || boxes.isEmpty)
                }
            }
            .padding()
            .navigationTitle("UI boxes")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } }
            }
        }
        .onAppear { detect() }
        .sheet(isPresented: $sharing) {
            if let share { ShareSheet(items: [share]) }
        }
    }

    /// The screenshot at its own shape (portrait or landscape), with every box drawn over it in the same space.
    private var picture: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                if let image {
                    let fitted = fit(image.size, in: geometry.size)
                    Image(uiImage: image).resizable().scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                    ForEach(Array(boxes.enumerated()), id: \.offset) { _, box in
                        let rect = CGRect(x: box.rect.minX * fitted.width, y: box.rect.minY * fitted.height,
                                          width: box.rect.width * fitted.width, height: box.rect.height * fitted.height)
                        Rectangle()
                            .stroke(box.score >= 0.6 ? Color.green : Color.yellow, lineWidth: 1.5)
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)
                        Text(box.label).font(.system(size: 8, weight: .semibold)).foregroundStyle(.black)
                            .padding(.horizontal, 2)
                            .background(box.score >= 0.6 ? Color.green : Color.yellow)
                            .position(x: rect.minX + 12, y: max(6, rect.minY - 6))
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "iphone.slash").font(.largeTitle)
                        Text("No frame yet").font(.headline)
                        Text("Start Screen Mirroring first.").font(.caption).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func fit(_ size: CGSize, in space: CGSize) -> CGSize {
        guard size.width > 0, size.height > 0 else { return space }
        let scale = min(space.width / size.width, space.height / size.height)
        return CGSize(width: size.width * scale, height: size.height * scale)
    }

    private func detect() {
        guard let frame = feed.frame else {
            note = "No frame: start Screen Mirroring first."
            return
        }
        busy = true
        source = frame
        image = UIImage(data: frame)
        let started = now()
        Task.detached(priority: .userInitiated) {
            let found = UIDetector.detect(frame)
            await MainActor.run {
                boxes = found
                busy = false
                note = String(format: "%.0f boxes in %.0f ms — %@", Double(found.count), (now() - started) * 1000,
                              UIDetector.isUsingModel ? "trained model" : "iOS Vision, no model pushed")
            }
        }
    }

    private func save() {
        guard let frame = source else { return }
        guard let boxed = UIDetector.annotated(frame, boxes: boxes) else { return }
        UIDetector.saveSample(frame, boxes: boxes)  // a training sample, images/ + labels/
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ui-boxes.jpg")
        guard (try? boxed.write(to: url)) != nil else { return }
        share = url
        sharing = true
    }
}
