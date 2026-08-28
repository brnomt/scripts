#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_replaygain.py
Aplica la ganancia ReplayGain DIRECTAMENTE a las muestras de audio, para
reproductores que ignoran las etiquetas (tags) ReplayGain.

Como funciona:
1. Lee la etiqueta REPLAYGAIN_TRACK_GAIN (o _ALBUM_) de cada archivo.
2. Mide el pico real del audio con `ffmpeg volumedetect`.
3. Aplica una ganancia LINEAL pura con el filtro `volume` de ffmpeg,
   limitando la ganancia positiva para que el pico NUNCA supere el techo
   (-headroom dBFS). No usa limitadores ni compresores: CERO clipping y
   CERO saturacion.
4. Reescribe el archivo y pone las etiquetas de ganancia a 0 dB para que el
   proceso sea IDEMPOTENTE (se puede re-ejecutar sin doble ganancia).

Formatos sin perdida (flac/wav/alac): re-encode sin perdida.
Formatos con perdida (mp3/aac/ogg/opus): se re-encodifican a bitrate alto.

Requisitos: ffmpeg y ffprobe en el PATH.
  Windows:  winget install Gyan.FFmpeg

Uso:
  python apply_replaygain.py [carpeta_o_archivo ...] [opciones]

Ejemplos:
  python apply_replaygain.py "F:/Musica" --dry-run
  python apply_replaygain.py "F:/Musica" --backup
  python apply_replaygain.py "F:/Musica" --jobs 4
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AUDIO_EXTS = {
    ".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".aifc", ".ape", ".wv", ".mpc", ".mka",
    ".alac", ".tta", ".dsf", ".dff",
}

DEFAULT_MIN_GAIN = 0.05  # dB


def require_ffmpeg():
    """Aborta si ffmpeg/ffprobe no estan en el PATH."""
    missing = [p for p in ("ffmpeg", "ffprobe") if not shutil.which(p)]
    if missing:
        sys.exit(
            "ERROR: no se encontro " + ", ".join(missing) +
            " en el PATH. Instala ffmpeg y anadelo al PATH.\n"
            "  Windows (winget): winget install Gyan.FFmpeg"
        )


def ffprobe_json(path):
    """Devuelve el JSON de ffprobe (streams + formato)."""
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format",
           "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "ffprobe error").strip()[:400])
    return json.loads(r.stdout)


def parse_db(value):
    """Convierte una cadena tipo '-7.23 dB' o '+2.0' a float. None si falla."""
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"(?i)\s*db\s*$", "", s).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def find_gains(tags):
    """Busca en un dict de tags las ganancias de pista y album (en dB)."""
    track = album = None
    for k, v in tags.items():
        kl = k.lower()
        # Solo etiquetas *_gain; descartar *_peak (que tambien contiene "gain").
        if "replaygain" not in kl or "peak" in kl:
            continue
        val = parse_db(v)
        if val is None:
            continue
        if "track" in kl:
            track = val
        elif "album" in kl:
            album = val
    return track, album


def find_peak_tag(tags, prefer="track"):
    """Busca REPLAYGAIN_*_PEAK (lineal; 1.0 = 0 dBFS, puede ser > 1)."""
    track = album = generic = None
    for k, v in tags.items():
        kl = k.lower()
        if "replaygain" not in kl or "peak" not in kl:
            continue
        try:
            peak = float(v)
        except (TypeError, ValueError):
            continue
        if peak <= 0.0:
            continue
        if "track" in kl:
            track = peak
        elif "album" in kl:
            album = peak
        else:
            generic = peak
    if prefer == "album":
        for cand in (album, track, generic):
            if cand is not None:
                return cand
    else:
        for cand in (track, album, generic):
            if cand is not None:
                return cand
    return None


def encoder_sample_fmt(codec, sample_fmt):
    """sample_fmt que el encoder acepta, o None para no forzar."""
    if not sample_fmt:
        return None
    fmt = sample_fmt.lower().rstrip("p")  # s16p -> s16
    c = codec.lower()
    if c == "flac" and fmt in ("s16", "s32"):
        return fmt
    if c == "alac" and fmt in ("s16", "s32"):
        return fmt + "p"  # alac encoder usa planar
    return None


def inspect(path):
    """Devuelve {'codec', 'sample_fmt', 'tags'} o None si no hay stream de audio."""
    data = ffprobe_json(path)
    audio = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio":
            audio = s
            break
    if audio is None:
        return None

    tags = {}
    fmt_tags = (data.get("format") or {}).get("tags") or {}
    tags.update(fmt_tags)
    for s in data.get("streams", []):
        t = s.get("tags") or {}
        tags.update(t)

    return {
        "codec": audio.get("codec_name", ""),
        "sample_fmt": audio.get("sample_fmt") or "",
        "tags": tags,
    }


def measure_peak_db(path):
    """Mide el pico real (max_volume en dBFS) del primer stream de audio."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
           "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"max_volume:\s*([-+]?\d+(?:\.\d+)?)\s*dB", r.stderr)
    if not m:
        raise RuntimeError("no se pudo medir el pico del audio")
    return float(m.group(1))


def codec_spec(codec, sample_fmt=None):
    """Argumentos de ffmpeg para re-encodar segun el codec.
    Devuelve None si el codec no se puede re-escribir con ffmpeg."""
    c = codec.lower()
    sf = encoder_sample_fmt(c, sample_fmt)

    # Sin perdida
    if c == "flac":
        spec = ["-c:a", "flac", "-compression_level", "8"]
        if sf:
            spec += ["-sample_fmt", sf]
        return spec
    if c.startswith("pcm_"):
        return ["-c:a", c]                 # conserva el mismo PCM exacto
    if c == "alac":
        spec = ["-c:a", "alac"]
        if sf:
            spec += ["-sample_fmt", sf]
        return spec
    if c == "wavpack":
        return ["-c:a", "wavpack"]

    # Con perdida (re-encode a alta calidad)
    if c == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", "320k"]
    if c == "aac":
        return ["-c:a", "aac", "-b:a", "256k"]
    if c == "vorbis":
        return ["-c:a", "libvorbis", "-q:a", "7"]
    if c == "opus":
        return ["-c:a", "libopus", "-b:a", "192k"]
    if c in ("wmav2", "wmapro"):
        return ["-c:a", "wmav2", "-b:a", "192k"]

    # ape/tta/...: ffmpeg puede decodificar pero no codificar.
    return None


def unique_backup_path(path):
    """path.bak, o path.bak.1 / .2 / ... si ya existe (no pisa el original)."""
    bak = Path(str(path) + ".bak")
    n = 1
    while bak.exists():
        bak = Path(str(path) + f".bak.{n}")
        n += 1
    return bak


def apply_gain(path, gain_db, spec, backup, new_peak_linear=None):
    """Aplica la ganancia re-encodando a un archivo temporal y lo reemplaza."""
    tmp = path.with_name(f"{path.stem}.rg_tmp{path.suffix}")
    peak_tag = "1.000000" if new_peak_linear is None else f"{new_peak_linear:.6f}"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-map", "0:a", "-map", "0:v?", "-map", "0:s?",
        "-map_metadata", "0",
        "-c", "copy",
        *spec,
        "-af", f"volume={gain_db:.4f}dB",
        "-metadata", "REPLAYGAIN_TRACK_GAIN=0.00 dB",
        "-metadata", "REPLAYGAIN_ALBUM_GAIN=0.00 dB",
        "-metadata", f"REPLAYGAIN_TRACK_PEAK={peak_tag}",
        "-metadata", f"REPLAYGAIN_ALBUM_PEAK={peak_tag}",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError((r.stderr or "ffmpeg error").strip()[:500])

    if not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError("ffmpeg no produjo salida")

    bak = None
    if backup:
        bak = unique_backup_path(path)
        os.replace(path, bak)
    try:
        os.replace(tmp, path)
    except Exception:
        if bak is not None and bak.exists() and not path.exists():
            os.replace(bak, path)
        raise


def process_file(path, args):
    """Devuelve (estado, detalle, ganancia_aplicada)."""
    path = Path(path)
    try:
        info = inspect(path)
        if info is None:
            return ("skip", "sin stream de audio", 0.0)

        track, album = find_gains(info["tags"])
        gain = track if args.mode == "track" else album
        if gain is None:
            gain = album if args.mode == "track" else track
        if gain is None:
            return ("skip", "sin etiqueta replaygain", 0.0)

        target = gain + args.preamp
        if abs(target) < args.min_gain:
            return ("skip", f"ya normalizado ({target:+.2f} dB)", target)

        peak = None
        if args.use_peak_tag:
            p = find_peak_tag(info["tags"], prefer=args.mode)
            if p is not None:
                peak = 20.0 * math.log10(p)
        if peak is None:
            peak = measure_peak_db(path)

        if target > 0:
            available = -peak - args.headroom
            applied = min(target, available)
            if applied <= 0:
                return ("skip",
                        f"sin headroom para subir sin clip (pico {peak:+.2f} dB)",
                        target)
        else:
            applied = target

        if abs(applied) < args.min_gain:
            return ("skip", f"cambio despreciable ({applied:+.2f} dB)", applied)

        spec = codec_spec(info["codec"], info.get("sample_fmt"))
        if spec is None:
            return ("skip",
                    f"codec no re-encodable con ffmpeg: {info['codec']}",
                    applied)

        detalle = (f"{applied:+.2f} dB  (tag {gain:+.2f} dB, "
                   f"pico {peak:+.2f} dB, techo {-args.headroom:+.2f} dB, "
                   f"codec {info['codec']})")

        if args.dry_run:
            return ("dry", detalle, applied)

        new_peak_lin = 10.0 ** ((peak + applied) / 20.0)
        apply_gain(path, applied, spec, args.backup, new_peak_linear=new_peak_lin)
        return ("ok", detalle, applied)

    except Exception as e:
        return ("error", str(e), 0.0)


def gather_files(paths, exts):
    """Recolecta archivos de audio de forma recursiva, sin duplicados."""
    found = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            if ".rg_tmp" in p.name:
                continue
            if p.suffix.lower() in exts:
                found.append(p)
        elif p.is_dir():
            for root, _dirs, names in os.walk(p):
                for n in names:
                    fp = Path(root) / n
                    if ".rg_tmp" in n:
                        # Resto de una ejecucion interrumpida: se puede borrar.
                        try:
                            fp.unlink()
                            print(f"  [limpia] temporal eliminado: {fp}",
                                  file=sys.stderr)
                        except OSError:
                            pass
                        continue
                    if fp.suffix.lower() in exts:
                        found.append(fp)
        else:
            print(f"  [aviso] no existe: {p}", file=sys.stderr)

    seen = set()
    uniq = []
    for f in found:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    return uniq


def main(argv=None):
    # Salida linea a linea aunque se redirija a un archivo (monitoreo en vivo).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description="Aplica la ganancia ReplayGain a las muestras de audio "
                    "(sin clipping ni saturacion).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("paths", nargs="*", default=["."],
                    help="carpetas o archivos (por defecto: carpeta actual)")
    ap.add_argument("--mode", choices=["track", "album"], default="track",
                    help="usar ganancia de pista (track) o de album (album). "
                         "Por defecto: track")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="techo de pico = -headroom dBFS para ganancia "
                         "positiva (default: 1.0 dB)")
    ap.add_argument("--preamp", type=float, default=0.0,
                    help="ganancia extra en dB aplicada a todo (default: 0)")
    ap.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN,
                    help="ignorar si |ganancia| < esto (default: 0.05 dB)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="archivos en paralelo (default: 1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo mostrar que se haria, sin modificar nada")
    ap.add_argument("--backup", action="store_true",
                    help="guardar el original como .bak antes de reemplazar")
    ap.add_argument("--use-peak-tag", action="store_true",
                    help="fiarse de la etiqueta REPLAYGAIN_*_PEAK para evitar "
                         "decodificar (mas rapido, algo menos preciso)")
    ap.add_argument("--ext", action="append", default=[],
                    help="anadir una extension (ej: --ext .m4b). Repetible.")
    args = ap.parse_args(argv)

    require_ffmpeg()

    exts = set(AUDIO_EXTS)
    for e in args.ext:
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        exts.add(e)

    files = gather_files(args.paths, exts)
    if not files:
        print("No se encontraron archivos de audio.")
        return 0

    print(f"Archivos encontrados: {len(files)}")
    if args.dry_run:
        print("MODO SIMULACION (--dry-run): no se modifica nada.")
    print("-" * 78)

    counts = {"ok": 0, "dry": 0, "skip": 0, "error": 0}
    label = {"ok": "OK  ", "dry": "DRY ", "skip": "SKIP", "error": "ERR "}

    executor = None
    if args.jobs <= 1:
        results = ((f, process_file(f, args)) for f in files)
    else:
        executor = ThreadPoolExecutor(max_workers=args.jobs)
        futures = {executor.submit(process_file, f, args): f for f in files}
        results = ((futures[fu], fu.result()) for fu in as_completed(futures))

    try:
        for f, (status, detail, _gain) in results:
            counts[status] = counts.get(status, 0) + 1
            print(f"[{label.get(status, status)}] {f}")
            if detail:
                print(f"        {detail}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    print("-" * 78)
    print(f"Resumen: {counts['ok']} aplicados, {counts['dry']} simulados, "
          f"{counts['skip']} omitidos, {counts['error']} errores")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(130)
