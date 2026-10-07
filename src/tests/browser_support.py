"""Use an installed headless browser; prefer Chrome over confined Snap Firefox."""
import shutil
import subprocess


def available():
    return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("firefox")


def launch(profile, url, width=390, height=844, output=subprocess.DEVNULL):
    browser = available()
    if not browser:
        raise RuntimeError("Chrome, Chromium or Firefox is required")
    if "firefox" in browser:
        args = [browser, "--headless", "--no-remote", "--profile", str(profile),
                "--window-size", f"{width},{height}", url]
    else:
        args = [browser, "--headless", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", f"--user-data-dir={profile}",
                f"--window-size={width},{height}", url]
    return subprocess.Popen(args, stdout=output, stderr=output)
