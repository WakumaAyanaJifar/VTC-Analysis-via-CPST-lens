"""Step 1: run BirdNET v2.4 on the annotated Sierra Nevada soundscapes (Zenodo 10.5281/zenodo.7050014).

Usage:  python run_birdnet.py <audio_dir> <species.csv> <out_csv>

Each recording is decoded completely into memory first and then passed to BirdNET, one file at a time.
Five recordings in this dataset (SNE_006, 008, 014, 023, 032) cannot be read by libsndfile
("flac decoder lost sync") even from the verified Zenodo zip. They are decoded with FFmpeg via PyAV
instead (pip install av), which reads all five completely. On a file both decoders can read,
the two give bit-identical samples. A file that neither decoder can read is reported and skipped.

Writes one row per 3-s window and species with confidence >= 0.10 (target list = the 56 annotated species).
The output is written file by file, so if the run is interrupted, running the same command again
continues where it stopped.
"""
import sys, csv, glob, os, time
import numpy as np
import soundfile as sf
import birdnet

try:
    import av  # FFmpeg-based decoder, used when libsndfile fails
except ImportError:
    av = None

MIN_CONF = 0.10


def load_model():
    # LiteRT (pip install ai-edge-litert) has wheels for Python 3.10-3.14 on Windows, macOS, and Linux.
    # TensorFlow is used only as a fallback (it has no Python 3.14 wheels).
    try:
        return birdnet.load("acoustic", "2.4", "tf", library="litert")
    except ValueError as e_litert:
        try:
            return birdnet.load("acoustic", "2.4", "tf")
        except ValueError:
            sys.exit("BirdNET could not load a model runtime. Install one with:\n"
                     "    python -m pip install ai-edge-litert\n"
                     "then run this script again.\n\nDetails: " + str(e_litert))


def decode_with_av(path):
    """Decode an audio file completely with FFmpeg (PyAV). Returns mono float32 in [-1, 1] and the sample rate."""
    scale = {"s16": 32768.0, "s16p": 32768.0, "s32": 2147483648.0, "s32p": 2147483648.0,
             "flt": 1.0, "fltp": 1.0, "dbl": 1.0, "dblp": 1.0}
    with av.open(path) as c:
        st = c.streams.audio[0]
        sr, chunks = st.rate, []
        for frame in c.decode(st):
            x = frame.to_ndarray().astype(np.float32) / scale[frame.format.name]
            chunks.append(x.mean(axis=0) if x.ndim == 2 and x.shape[0] > 1 else x.reshape(-1))
    return np.concatenate(chunks), sr


def read_audio(path):
    """Try libsndfile first; fall back to FFmpeg (PyAV). Returns (audio, sr, decoder_name)."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr, "libsndfile"
    except Exception as e_sf:
        if av is None:
            raise RuntimeError(f"{e_sf} (install PyAV with 'python -m pip install av' to decode it with FFmpeg)")
        audio, sr = decode_with_av(path)
        return audio, sr, "ffmpeg"


def main():
    audio_dir, species_csv, out_csv = sys.argv[1:4]

    model = load_model()
    sci2label = {s.split("_")[0]: s for s in model.species_list}
    rows = list(csv.DictReader(open(species_csv, encoding="utf-8-sig")))
    label2code = {sci2label[r["Scientific Name"]]: r["Species eBird Code"] for r in rows}

    files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True))
    print(len(files), "files;", len(label2code), "target species", flush=True)
    if not files:
        sys.exit(f"No .flac files found under '{audio_dir}'. Unzip soundscape_data.zip into that folder first.")

    # Resume support: skip files already present in the output
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            done = {r["filename"] for r in csv.DictReader(f)}
        done_marker = out_csv + ".done"
        if os.path.exists(done_marker):
            done |= set(open(done_marker, encoding="utf-8").read().split())
    new_file = not os.path.exists(out_csv)
    failed = []

    with open(out_csv, "a", newline="", encoding="utf-8") as f, \
         open(out_csv + ".done", "a", encoding="utf-8") as fdone:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filename", "start_s", "end_s", "ebird_code", "confidence"])
        for i, path in enumerate(files, 1):
            name = os.path.basename(path)
            if name in done:
                print(f"[{i}/{len(files)}] {name}: already done, skipped", flush=True)
                continue
            t0 = time.time()
            try:
                audio, sr, decoder = read_audio(path)
            except Exception as e:
                print(f"[{i}/{len(files)}] {name}: COULD NOT DECODE ({e}); skipped", flush=True)
                failed.append((name, str(e)))
                continue
            if decoder != "libsndfile":
                with open(out_csv + ".decoder_log.txt", "a", encoding="utf-8") as fl:
                    fl.write(f"{name}\t{decoder}\t{len(audio) / sr / 60:.2f} min\n")
            res = model.predict_arrays((audio, sr), top_k=None, default_confidence_threshold=MIN_CONF,
                                       custom_species_list=list(label2code), overlap_duration_s=0,
                                       n_workers=2, batch_size=16)
            arr = res.to_structured_array()
            for r in arr:
                w.writerow([name, float(r["start_time"]), float(r["end_time"]),
                            label2code[str(r["species_name"])], round(float(r["confidence"]), 5)])
            f.flush()
            fdone.write(name + "\n"); fdone.flush()
            print(f"[{i}/{len(files)}] {name}: {len(arr)} detections ({time.time() - t0:.0f} s; decoder {decoder})", flush=True)

    if not failed and os.path.exists(out_csv + ".failed.txt"):
        os.remove(out_csv + ".failed.txt")
    if failed:
        with open(out_csv + ".failed.txt", "w", encoding="utf-8") as f:
            for n, e in failed:
                f.write(f"{n}\t{e}\n")
        print(f"\n{len(failed)} file(s) could not be decoded; listed in {out_csv}.failed.txt")
        print("Install PyAV (python -m pip install av) and run the same command again; only these files will be processed.")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
