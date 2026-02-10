# print2pages (macOS) — Chunked Printing With Delays

This note explains the `print2pages` script, what it does, and how to use it.

## What it does
`print2pages` prints a PDF in **2-page chunks** with a pause between chunks.
This reduces paper‑jam risk on printers that misfeed when multiple pages are sent quickly.

It works by:
- Reading how many pages are in the PDF.
- Printing pages 1–2, waiting a few seconds, printing 3–4, and so on.
- Inheriting your **printer defaults** (duplex, color, paper size) unless you explicitly override them.

## Requirements
- macOS with the built-in `lp` printing system (CUPS).
- A PDF file to print.

## Installation
Create a `bin` folder and save the script:

```zsh
mkdir -p "$HOME/bin"

cat <<'EOF' > "$HOME/bin/print2pages"
#!/bin/zsh
set -euo pipefail

file="${1:-}"
delay="${2:-6}"
printer="${3:-}"

if [[ -z "$file" ]]; then
  echo "Usage: print2pages /path/to/file.pdf [delay_seconds] [printer_name]" >&2
  exit 1
fi

if [[ -z "$printer" ]]; then
  printer=$(lpstat -d | sed 's/^system default destination: //')
fi

pages=$(mdls -name kMDItemNumberOfPages -raw "$file" 2>/dev/null || true)
if [[ ! "$pages" =~ ^[0-9]+$ ]]; then
  echo "Could not determine PDF page count: $file" >&2
  exit 1
fi

start=1
while [[ $start -le $pages ]]; do
  end=$((start+1))
  if [[ $end -gt $pages ]]; then end=$pages; fi
  lp -d "$printer" -o page-ranges="${start}-${end}" "$file"
  sleep "$delay"
  start=$((start+2))
done
EOF

chmod +x "$HOME/bin/print2pages"
```

## Usage
Print a PDF with an 8‑second delay between each 2‑page chunk:

```zsh
"$HOME/bin/print2pages" "$HOME/Desktop/yourfile.pdf" 8
```

If you want to target a specific printer:

```zsh
"$HOME/bin/print2pages" "$HOME/Desktop/yourfile.pdf" 8 "Canon MF632C/634C"
```

### Tip: drag‑and‑drop
In Terminal, you can drag a PDF from Finder into the command to insert its path.

## Printer defaults (duplex, paper, color)
The script **inherits** whatever the printer defaults are.
If you want to force duplex inside the script, replace the `lp` line with:

```zsh
lp -d "$printer" -o sides=two-sided-long-edge -o page-ranges="${start}-${end}" "$file"
```

## Finding your printer name
```zsh
lpstat -p
lpstat -d
```

## Troubleshooting
- "Could not determine PDF page count": the file may not be a PDF or is damaged.
- If jobs are still too fast, increase the delay (e.g. `10` or `12` seconds).
- If nothing prints, verify the printer name from `lpstat -p`.

## Why use $HOME instead of ~
On some keyboards, the tilde key inserts a **different** character (not ASCII `~`).
`$HOME` always works.
