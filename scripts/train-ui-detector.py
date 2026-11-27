"""Trains the UI box detector and exports it for the phone.

The phone writes samples itself: Diagnostics › UI boxes over the screen › Save puts a screenshot in
Documents/uidataset/images and its boxes in Documents/uidataset/labels (one `0 cx cy w h` line per box,
normalised), the same layout SA-Ai-iOS trained its detector with. Pull that folder off the phone, then:

    python -m pip install ultralytics torch coremltools
    python scripts/train-ui-detector.py ~/Downloads/uidataset

It splits train/val, trains a single-class YOLO (`ui`), exports Core ML, and prints the file to push and
where to push it. The phone reads `Documents/uidetector.mlpackage` (or .mlmodelc / .mlmodel) at launch of
the box sheet: no rebuild, just the file.

Why single class: the job is "box what is on the screen", not "name it". One class keeps the model small,
lets NMS stay simple, and needs far less labelled data than a button/field/icon taxonomy.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRACTION_VAL = 0.2


def build(folder: Path) -> Path:
    images, labels = folder / "images", folder / "labels"
    shots = sorted(p for p in images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not shots:
        sys.exit(f"no screenshots in {images}; press Save in the app's box sheet first")

    data = folder / "yolo"
    if data.exists():
        shutil.rmtree(data)
    for split in ("train", "val"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)

    cut = max(1, int(len(shots) * (1 - FRACTION_VAL)))
    for index, shot in enumerate(shots):
        split = "train" if index < cut else "val"
        box = labels / (shot.stem + ".txt")
        if not box.exists():
            continue
        shutil.copy(shot, data / "images" / split / shot.name)
        shutil.copy(box, data / "labels" / split / box.name)

    yaml = data / "dataset.yaml"
    yaml.write_text(
        f"path: {data}\ntrain: images/train\nval: images/val\n\nnames:\n    0: ui\n", encoding="utf-8")
    print(f"{len(shots)} samples → {yaml}")
    return yaml


def train(yaml: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics torch first")

    model = YOLO("yolo11n.pt")
    model.train(data=str(yaml), epochs=60, imgsz=640, batch=16, single_cls=True,
                project=str(yaml.parent.parent / "runs"), name="ui_detector", exist_ok=True)
    best = Path(model.trainer.best)
    print(f"weights: {best}")
    # nms=False keeps the raw (1, C, N) head the phone decodes; a Core ML export with NMS bakes the boxes
    # into a variable number of outputs, which the app's parser does not read.
    exported = YOLO(str(best)).export(format="coreml", imgsz=640, nms=False)
    print(f"coreml: {exported}")
    return Path(exported)


def main() -> None:
    folder = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT / "uidataset"
    yaml = build(folder)
    model = train(yaml)
    print("\npush it (Python on the phone's side, same helper the model uses):")
    print(f"  python scripts/push-model.py \\\"{model}\\\" --remote-name uidetector.mlpackage")


if __name__ == "__main__":
    main()
