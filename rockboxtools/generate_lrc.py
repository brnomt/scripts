#!/usr/bin/env python3
"""
generate_lrc.py — Escanea carpetas de música y genera archivos .lrc con letras
sincronizadas desde LRCLIB (lrclib.net), optimizado para iPod + Rockbox.

Estructura esperada:
    MUSIC / {Artist} - {Album} / Song.flac (o .wav)

Si no puede interpretar artista/álbum del nombre de carpeta, extrae los
metadatos del archivo de audio (tags Vorbis de FLAC o RIFF INFO de WAV).
Si aún así falla, saltea la canción.

Dependencias opcionales (mejoran la lectura de metadatos):
    pip install mutagen

Uso:
    python generate_lrc.py [directorio_raíz]

    Si no se especifica, se usa "MUSIC" en el directorio actual.
    Para un iPod montado:  python generate_lrc.py "E:\\MUSIC"
"""

import os
import re
import sys
import json
import struct
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Optional

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

LRCLIB_BASE = "https://lrclib.net/api"
REQUEST_DELAY = 1.5      # segundos entre requests
REQUEST_TIMEOUT = 15      # segundos de timeout HTTP
AUDIO_EXTENSIONS = {".flac", ".wav"}

# ─── METADATOS DE AUDIO ──────────────────────────────────────────────────────

def _read_flac_vorbis_comments(filepath: str) -> Optional[dict]:
    """
    Lee los tags Vorbis Comment de un archivo FLAC sin dependencias externas.
    Formato FLAC:
        4 bytes: "fLaC"
        luego bloques de metadatos:
            1 byte:  flags (bit 7=last, bits 0-6=type)
            3 bytes: block length (big-endian)
            N bytes: block data
    El bloque Vorbis Comment es type=4.
    """
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
            if header != b"fLaC":
                return None

            while True:
                block_header = f.read(4)
                if len(block_header) < 4:
                    break

                is_last = (block_header[0] & 0x80) != 0
                block_type = block_header[0] & 0x7F
                block_len = (block_header[1] << 16) | (block_header[2] << 8) | block_header[3]

                if block_type == 4:  # VORBIS_COMMENT
                    block_data = f.read(block_len)
                    return _parse_vorbis_comment_block(block_data)

                if is_last:
                    break
                f.seek(block_len, 1)  # skip this block

    except (IOError, OSError, struct.error):
        pass

    return None


def _parse_vorbis_comment_block(data: bytes) -> dict:
    """
    Parsea un bloque Vorbis Comment.
    - 4 bytes LE: vendor length
    - vendor string
    - 4 bytes LE: user comment count
    - por cada comentario:
        - 4 bytes LE: comment length
        - comment string "KEY=VALUE"
    """
    result: dict[str, str] = {}
    offset = 0

    def read_u32_le(b: bytes, pos: int) -> tuple[int, int]:
        if pos + 4 > len(b):
            return 0, len(b)
        val = struct.unpack_from("<I", b, pos)[0]
        return val, pos + 4

    if len(data) < 4:
        return result

    vendor_len, offset = read_u32_le(data, offset)
    offset += vendor_len  # skip vendor string

    if offset + 4 > len(data):
        return result

    comment_count, offset = read_u32_le(data, offset)

    for _ in range(comment_count):
        if offset + 4 > len(data):
            break
        comment_len, offset = read_u32_le(data, offset)
        if offset + comment_len > len(data):
            break
        raw = data[offset:offset + comment_len].decode("utf-8", errors="replace")
        offset += comment_len

        if "=" in raw:
            key, value = raw.split("=", 1)
            key_upper = key.upper().strip()
            val = value.strip()
            # Solo guardamos la primera ocurrencia de cada campo
            if key_upper not in result and val:
                result[key_upper] = val

    return result


def _read_wav_riff_info(filepath: str) -> Optional[dict]:
    """
    Lee chunks RIFF INFO de un archivo WAV sin dependencias externas.
    Busca chunks 'LIST' con sub-type 'INFO' que contienen tags como IART, INAM, IPRD.
    """
    try:
        with open(filepath, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                return None
            f.read(4)  # file size
            wave = f.read(4)
            if wave != b"WAVE":
                return None

            result: dict[str, str] = {}
            tag_map = {
                "IART": "ARTIST",
                "INAM": "TITLE",
                "IPRD": "ALBUM",
                "IGNR": "GENRE",
                "ICRD": "DATE",
                "ITRK": "TRACKNUMBER",
            }

            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break

                chunk_size_raw = f.read(4)
                if len(chunk_size_raw) < 4:
                    break
                chunk_size = struct.unpack_from("<I", chunk_size_raw, 0)[0]

                if chunk_id == b"LIST":
                    list_type = f.read(4)
                    if list_type == b"INFO":
                        end_pos = f.tell() + chunk_size - 4
                        while f.tell() < end_pos:
                            tag_id = f.read(4)
                            if len(tag_id) < 4:
                                break
                            tag_size_raw = f.read(4)
                            if len(tag_size_raw) < 4:
                                break
                            tag_size = struct.unpack_from("<I", tag_size_raw, 0)[0]
                            tag_data = f.read(tag_size).rstrip(b"\x00")
                            tag_str = tag_id.decode("ascii", errors="replace")
                            mapped = tag_map.get(tag_str, tag_str)
                            value = tag_data.decode("utf-8", errors="replace").strip()
                            if mapped not in result and value:
                                result[mapped] = value
                    else:
                        f.seek(chunk_size - 4, 1)
                else:
                    f.seek(chunk_size, 1)

            return result if result else None

    except (IOError, OSError, struct.error):
        pass

    return None


def read_audio_metadata(filepath: str) -> Optional[dict]:
    """
    Lee metadatos (artist, album, title) del archivo de audio.
    Intenta por orden:
        1. mutagen (si está instalado) — soporta FLAC, WAV, MP3, etc.
        2. Parser manual de Vorbis Comments (FLAC)
        3. Parser manual de RIFF INFO (WAV)
    Retorna dict con claves 'artist', 'album', 'title' o None.
    """
    # ── 1. mutagen (más fiable) ──────────────────────────────────────
    try:
        import mutagen
        audio = mutagen.File(filepath)
        if audio is not None:
            artist = _first_tag(audio, "artist", "ARTIST", "TPE1", "©ART")
            album = _first_tag(audio, "album", "ALBUM", "TALB", "©alb")
            title = _first_tag(audio, "title", "TITLE", "TIT2", "©nam")

            if artist or album or title:
                return {
                    "artist": artist or "",
                    "album": album or "",
                    "title": title or "",
                }
    except ImportError:
        pass
    except Exception:
        pass

    # ── 2. Parser manual ─────────────────────────────────────────────
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".flac":
        vorbis = _read_flac_vorbis_comments(filepath)
        if vorbis:
            return {
                "artist": vorbis.get("ARTIST", ""),
                "album": vorbis.get("ALBUM", ""),
                "title": vorbis.get("TITLE", ""),
            }

    if ext == ".wav":
        riff = _read_wav_riff_info(filepath)
        if riff:
            return {
                "artist": riff.get("ARTIST", ""),
                "album": riff.get("ALBUM", ""),
                "title": riff.get("TITLE", ""),
            }

    return None


def _first_tag(audio, *keys: str) -> str:
    """Extrae el primer valor de una lista de tags posibles (mutagen)."""
    for key in keys:
        val = audio.get(key)
        if val:
            if isinstance(val, list):
                return str(val[0])
            return str(val)
    return ""


# ─── UTILIDADES ───────────────────────────────────────────────────────────────

def simplify_track_name(name: str) -> str:
    """
    Simplifica el nombre de una canción eliminando sufijos que
    LRCLIB no suele indexar bien.
    """
    name = re.sub(r'\s*\(feat\.?\s[^)]+\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(ft\.?\s[^)]+\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(with\s[^)]+\)', '', name, flags=re.IGNORECASE)

    name = re.sub(
        r'\s+-\s+(Remastered|Live|Bonus Track|Radio Edit|Extended|'
        r'Acoustic|Demo|Instrumental|Single Version|Album Version|'
        r'Original Mix|Club Mix|Edit|Version)\b.*$',
        '', name, flags=re.IGNORECASE,
    )

    candidate = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
    if len(candidate) >= 3:
        name = candidate

    if " - " in name:
        parts = name.split(" - ", 1)
        if len(parts[1]) < 30:
            name = parts[0]

    return name.strip()


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """
    Extrae artista y álbum de un nombre de carpeta.
    "Artist - Album"         → (Artist, Album)
    "Artist - Album (2024)"  → (Artist, Album)
    "Artist"                 → (Artist, "")   ← ambiguo, se intentará metadata
    """
    folder_name = folder_name.strip()

    if " - " in folder_name:
        artist, album = folder_name.split(" - ", 1)
        artist = artist.strip()
        album = album.strip()
        album_clean = re.sub(r'\s*[\[\(]\d{4}[\]\)]\s*', '', album).strip()
        return artist, album_clean

    # Sin " - " no podemos separar → devolvemos solo el artista
    # y marcamos el álbum como vacío (luego se intentará metadata)
    return folder_name, ""


def parse_track_name(filename: str) -> str:
    """
    Extrae el título del nombre del archivo (sin extensión ni track number).
    """
    name = os.path.splitext(filename)[0].strip()

    # Si es "Artist - Song", tomar solo la parte derecha
    if " - " in name:
        parts = name.split(" - ")
        if re.match(r'^\d{1,3}$', parts[0].strip()):
            name = parts[-1].strip()
        elif len(parts) >= 2:
            name = parts[-1].strip()

    # Quitar número de track
    name = re.sub(r'^\d{1,3}\.\s*', '', name)
    name = re.sub(r'^\d{1,3}\s*[-–—]\s*', '', name)
    name = re.sub(r'^\d{1,3}\s{1,3}', '', name)

    return name.strip()


# ─── API LRCLIB ───────────────────────────────────────────────────────────────

def fetch_lrclib_exact(artist: str, track: str, album: str = "",
                       duration: int = 0) -> Optional[dict]:
    """GET /api/get — búsqueda exacta."""
    params = {"artist_name": artist, "track_name": track}
    if album:
        params["album_name"] = album
    if duration > 0:
        params["duration"] = str(duration)

    url = f"{LRCLIB_BASE}/get?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LRC-Generator/2.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    HTTP {e.code}")
    except Exception as e:
        print(f"    Error de red: {e}")

    return None


def fetch_lrclib_search(artist: str, track: str) -> Optional[list[dict]]:
    """GET /api/search — búsqueda fuzzy."""
    params = {"artist_name": artist, "track_name": track}
    url = f"{LRCLIB_BASE}/search?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LRC-Generator/2.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    Error de red (search): {e}")

    return None


# ─── CONVERSIÓN A LRC ────────────────────────────────────────────────────────

def lrclib_to_lrc(data: dict, track_name: str, artist_name: str) -> Optional[str]:
    """Convierte respuesta JSON de LRCLIB a contenido .lrc."""
    synced = data.get("syncedLyrics", "")
    plain = data.get("plainLyrics", "")
    lyrics_text = synced or plain

    if not lyrics_text or not lyrics_text.strip():
        return None

    lines = [
        f"[ti:{track_name}]",
        f"[ar:{artist_name}]",
        f"[by:LRCLIB Generator ({'SYNCED' if synced else 'UNSYNCED'})]",
        "",
    ]
    for line in lyrics_text.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    return "\n".join(lines) + "\n"


def pick_best_result(results: list[dict]) -> Optional[dict]:
    """Elige el mejor resultado: synced > plain > primero."""
    best_synced = None
    best_plain = None
    for r in results:
        if r.get("syncedLyrics") and best_synced is None:
            best_synced = r
        if r.get("plainLyrics") and best_plain is None:
            best_plain = r
        if best_synced:
            break
    return best_synced or best_plain or (results[0] if results else None)


# ─── PROCESAMIENTO DE UNA CANCIÓN ────────────────────────────────────────────

def process_track(audio_path: str, artist: str, album: str) -> bool:
    """
    Procesa un archivo de audio: busca letras y genera el .lrc.
    Retorna True si se generó (o ya existía), False si no se encontraron letras.
    """
    folder = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)
    track_name = parse_track_name(filename)

    lrc_filename = os.path.splitext(filename)[0] + ".lrc"
    lrc_path = os.path.join(folder, lrc_filename)

    # ── Skip si ya existe ────────────────────────────────────────────
    if os.path.exists(lrc_path):
        print(f"  ⏭  {track_name}  →  .lrc ya existe")
        return True

    print(f"  🔍 {track_name}  →  buscando…")

    strategy = _search_lyrics(artist, track_name, album)

    if strategy:
        lrc = lrclib_to_lrc(strategy["data"], track_name, strategy["effective_artist"])
        if lrc:
            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc)
            synced = "✓ synced" if strategy["data"].get("syncedLyrics") else "⚠ unsynced"
            print(f"  ✅ {track_name}  →  {synced}  ({strategy['source']})")
            return True

    print(f"  ❌ {track_name}  →  sin letras")
    return False


def _search_lyrics(artist: str, track: str, album: str) -> Optional[dict]:
    """
    Cascada de búsqueda (misma lógica que el Go de SpotiFLAC).
    Retorna {"data": ..., "effective_artist": ..., "source": ...} o None.
    """

    def try_data(d: Optional[dict], src: str) -> Optional[dict]:
        if d and (d.get("syncedLyrics") or d.get("plainLyrics")):
            return {"data": d, "effective_artist": artist, "source": src}
        return None

    # 1. Exacta con álbum
    r = try_data(fetch_lrclib_exact(artist, track, album), "exacta + álbum")
    if r: return r

    # 2. Exacta sin álbum
    if album:
        r = try_data(fetch_lrclib_exact(artist, track), "exacta s/álbum")
        if r: return r

    # 3. Fuzzy search
    results = fetch_lrclib_search(artist, track)
    if results:
        best = pick_best_result(results)
        r = try_data(best, "search")
        if r: return r

    # 4. Simplificar título y reintentar
    simplified = simplify_track_name(track)
    if simplified != track and len(simplified) >= 3:
        r = try_data(fetch_lrclib_exact(artist, simplified, album), "simplificado")
        if r: return r

        results = fetch_lrclib_search(artist, simplified)
        if results:
            best = pick_best_result(results)
            r = try_data(best, "simplificado + search")
            if r: return r

    return None


# ─── ESCANEO PRINCIPAL ───────────────────────────────────────────────────────

def scan_and_process(root_dir: str) -> None:
    """
    Escanea recursivamente, extrayendo artista/álbum de la carpeta
    o de los metadatos del archivo si la carpeta no es clara.
    """
    root = os.path.abspath(root_dir)

    if not os.path.isdir(root):
        print(f'❌ Error: "{root}" no es un directorio válido.')
        sys.exit(1)

    # ── Recolectar archivos ──────────────────────────────────────────
    audio_files: list[dict] = []

    print(f"🔎 Escaneando: {root}")
    print(f"   Extensiones: {', '.join(AUDIO_EXTENSIONS)}")
    print()

    for dirpath, dirnames, filenames in os.walk(root):
        folder_name = os.path.basename(dirpath)

        # Saltar carpetas de sistema
        if folder_name.startswith(".") or folder_name in (
            "$RECYCLE.BIN", "System Volume Information",
        ):
            continue

        folder_artist, folder_album = parse_folder_name(folder_name)

        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                full_path = os.path.join(dirpath, f)
                audio_files.append({
                    "path": full_path,
                    "folder_artist": folder_artist,
                    "folder_album": folder_album,
                    "folder_name": folder_name,
                })

    total = len(audio_files)
    if total == 0:
        print("❌ No se encontraron archivos .flac o .wav.")
        return

    print(f"🎵 {total} canciones encontradas.\n")

    # ── Procesar ─────────────────────────────────────────────────────
    ok = 0
    skipped_meta = 0
    no_lyrics = 0

    for i, item in enumerate(audio_files, 1):
        path = item["path"]
        folder = item["folder_name"]
        artist = item["folder_artist"]
        album = item["folder_album"]

        # ── Resolver artista si es ambiguo ──────────────────────────
        # Caso 1: carpeta sin " - " → no sabemos si es artista o álbum
        # Caso 2: artista vacío
        # En ambos casos, intentamos leer metadatos del archivo
        used_metadata = False
        artist_from_meta = ""
        album_from_meta = ""

        if not artist or not album or " - " not in folder:
            meta = read_audio_metadata(path)
            if meta:
                artist_from_meta = meta.get("artist", "")
                album_from_meta = meta.get("album", "")
                if artist_from_meta or album_from_meta:
                    used_metadata = True
                    # Si la carpeta ya dio un artista, lo mantenemos como fallback
                    if not artist:
                        artist = artist_from_meta
                    if not album:
                        album = album_from_meta

        # Si después de todo no hay artista, saltamos
        if not artist:
            print(f"[{i}/{total}] 📁 {folder}")
            track_name = parse_track_name(os.path.basename(path))
            if used_metadata:
                print(f"  ⚠  {track_name}  →  metadatos insuficientes, saltando")
            else:
                print(f"  ⚠  {track_name}  →  no se pudo determinar el artista, saltando")
            skipped_meta += 1
            continue

        # ── Indicar fuente de metadatos ─────────────────────────────
        print(f"[{i}/{total}] 📁 {folder}")
        if used_metadata:
            src_parts = []
            if artist_from_meta and artist == artist_from_meta:
                src_parts.append("artista")
            if album_from_meta and album == album_from_meta:
                src_parts.append("álbum")
            if src_parts:
                print(f"     📋 metadatos → {', '.join(src_parts)}: {artist} — {album}")

        # ── Procesar ────────────────────────────────────────────────
        lrc_path = os.path.splitext(path)[0] + ".lrc"
        existed = os.path.exists(lrc_path)
        success = process_track(path, artist, album)

        if success:
            ok += 1
        else:
            if not existed:
                no_lyrics += 1

        if i < total:
            time.sleep(REQUEST_DELAY)

    # ── Resumen ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("📊 RESUMEN")
    print(f"   Total canciones:        {total}")
    print(f"   .lrc creados / ya tenía: {ok}")
    print(f"   Sin letras encontradas:  {no_lyrics}")
    print(f"   Saltados (sin artista):  {skipped_meta}")
    print(f"   Fuente: LRCLIB (lrclib.net)")
    print("=" * 60)

    if skipped_meta > 0:
        print()
        print("💡 Algunas canciones se saltaron porque no se pudo determinar")
        print("   el artista ni de la carpeta ni de los metadatos internos.")
        print("   Instalá mutagen para mejor soporte:  pip install mutagen")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        music_root = sys.argv[1]
    else:
        music_root = os.path.join(os.getcwd(), "MUSIC")

    print("╔══════════════════════════════════════════════╗")
    print("║   🎵 LRC Generator para iPod + Rockbox v2   ║")
    print("║   Fuente: LRCLIB (lrclib.net)               ║")
    print("║   Metadatos: mutagen o parser nativo        ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    scan_and_process(music_root)
