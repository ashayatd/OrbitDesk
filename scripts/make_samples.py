"""Create demo inputs for /image and /voice in evals/samples/.

- vpn_error.png, bluescreen.png: fake error dialogs rendered with Pillow
- *.wav: spoken questions via Windows' built-in speech synthesiser (System.Speech);
  skipped on other platforms (the committed WAVs are reused).

    uv run python scripts/make_samples.py
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "evals" / "samples"

DIALOGS = {
    "vpn_error.png": (
        "OrbitConnect VPN",
        "Error VPN-809: Gateway unreachable",
        "Could not connect to vpn.orbit.example",
    ),
    "bluescreen.png": ("Windows", "Stop code: 0x0000007E", "SYSTEM_THREAD_EXCEPTION_NOT_HANDLED"),
}
SPEECH = {
    "annual_leave.wav": "How many days of annual leave do I get each year?",
    "vpn_hotel.wav": "My VPN shows error eight oh nine at the hotel. What should I do?",
}


def font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def dialog(title: str, line1: str, line2: str, path: Path) -> None:
    img = Image.new("RGB", (640, 260), "#f3f3f3")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 640, 44], fill="#1f4e79")
    d.text((16, 10), title, fill="white", font=font(20))
    d.ellipse([28, 80, 76, 128], fill="#c42b1c")
    d.text((45, 86), "!", fill="white", font=font(30))
    d.text((100, 82), line1, fill="#111111", font=font(22))
    d.text((100, 122), line2, fill="#333333", font=font(16))
    d.rectangle([520, 200, 620, 236], outline="#666666", fill="#e1e1e1")
    d.text((555, 207), "OK", fill="#111111", font=font(16))
    img.save(path)


def speak(text: str, path: Path) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in DIALOGS.items():
        dialog(*lines, OUT / name)
        print("wrote", name)
    if sys.platform == "win32":
        for name, text in SPEECH.items():
            speak(text, OUT / name)
            print("wrote", name)


if __name__ == "__main__":
    main()
