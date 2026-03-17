# print2pages (macOS) — Chunked Printing With Delays

This note explains the working setup with two scripts:
`print2pages` (direct use) and `print2pages-auto` (Automator).

## What it does
`print2pages` prints a PDF in **2-page chunks** with a pause between chunks.
This reduces paper‑jam risk on printers that misfeed when multiple pages are sent quickly.

It works by:
- Reading how many pages are in the PDF.
- Printing pages 1–2, waiting a few seconds, printing 3–4, and so on.
- Inheriting your **printer defaults** (duplex, color, paper size) unless you explicitly override them.

The Automator variant (`print2pages-auto`) adds:
- Waiting briefly for the PDF to finish writing.
- Copying the temp PDF into a user cache folder (important for Automator print workflows).
- Extra page‑count fallbacks.

## Requirements
- macOS with the built-in `lp` printing system (CUPS).
- A PDF file to print.

## Installation
Create a `bin` folder and save both scripts:

```zsh
mkdir -p "$HOME/bin"

cat <<'EOF' > "$HOME/bin/print2pages"
#!/bin/zsh
set -euo pipefail

file="${1:-}"
delay="${2:-2}"
printer="${3:-}"
wait_for_job="${WAIT_FOR_JOB:-1}"

if [[ -z "$file" ]]; then
  echo "Usage: print2pages /path/to/file.pdf [delay_seconds] [printer_name]" >&2
  exit 1
fi

if [[ ! -f "$file" ]]; then
  echo "File not found: $file" >&2
  exit 1
fi

if [[ -z "$printer" ]]; then
  printer=$(lpstat -d | sed 's/^system default destination: //')
fi

if [[ -z "$printer" ]]; then
  echo "No default printer set. Choose one in System Settings → Printers & Scanners." >&2
  exit 1
fi

wait_enabled=1
case "${wait_for_job:l}" in
  0|no|false|off) wait_enabled=0 ;;
esac

job_in_queue() {
  local job_id="$1"
  lpstat -W not-completed -o "$printer" 2>/dev/null | awk '{print $1}' | grep -qx "$job_id"
}

wait_for_job_completion() {
  local job_id="$1"
  [[ -z "$job_id" ]] && return 0
  while job_in_queue "$job_id"; do
    sleep 1
  done
}

get_pages() {
  local f="$1"
  local pages=""
  pages=$(mdls -name kMDItemNumberOfPages -raw "$f" 2>/dev/null || true)
  if [[ "$pages" =~ ^[0-9]+$ ]]; then
    echo "$pages"
    return 0
  fi
  pages=$(sips -g pageCount "$f" 2>/dev/null | awk '/pageCount/ {print $2; exit}')
  if [[ "$pages" =~ ^[0-9]+$ ]]; then
    echo "$pages"
    return 0
  fi
  pages=$(/usr/bin/osascript - "$f" <<'APPLESCRIPT'
use framework "PDFKit"
use framework "Foundation"
use scripting additions
on run argv
  set p to item 1 of argv
  set theURL to current application's |NSURL|'s fileURLWithPath:p
  set thePDF to current application's PDFDocument's alloc()'s initWithURL:theURL
  if thePDF is not missing value then
    return (thePDF's pageCount() as integer) as string
  end if
end run
APPLESCRIPT
)
  pages=$(echo "$pages" | tr -d '[:space:]')
  if [[ "$pages" =~ ^[0-9]+$ ]]; then
    echo "$pages"
    return 0
  fi
  if command -v pdfinfo >/dev/null 2>&1; then
    pages=$(pdfinfo "$f" 2>/dev/null | awk '/^Pages:/ {print $2; exit}')
    if [[ "$pages" =~ ^[0-9]+$ ]]; then
      echo "$pages"
      return 0
    fi
  fi
  return 1
}

pages=$(get_pages "$file" || true)
if [[ ! "$pages" =~ ^[0-9]+$ ]]; then
  echo "Could not determine PDF page count: $file" >&2
  exit 1
fi

start=1
while [[ $start -le $pages ]]; do
  end=$((start+1))
  if [[ $end -gt $pages ]]; then end=$pages; fi
  job_out=$(lp -d "$printer" -o page-ranges="${start}-${end}" "$file")
  echo "$job_out"
  job_id=$(echo "$job_out" | awk '{print $4}')
  if [[ $wait_enabled -eq 1 ]]; then
    wait_for_job_completion "$job_id"
  fi
  sleep "$delay"
  start=$((start+2))
done
EOF

chmod +x "$HOME/bin/print2pages"
```

Automator‑only script:

```zsh
cat <<'EOF' > "$HOME/bin/print2pages-auto"
#!/bin/zsh
set -euo pipefail

file="${1:-}"
delay="${2:-2}"
printer="${3:-}"
wait_for_job="${WAIT_FOR_JOB:-1}"

if [[ -z "$file" ]]; then
  echo "Usage: print2pages-auto /path/to/file.pdf [delay_seconds] [printer_name]" >&2
  exit 1
fi

if [[ -z "$printer" ]]; then
  printer=$(lpstat -d | sed 's/^system default destination: //')
fi

if [[ -z "$printer" ]]; then
  echo "No default printer set. Choose one in System Settings → Printers & Scanners." >&2
  exit 1
fi

wait_enabled=1
case "${wait_for_job:l}" in
  0|no|false|off) wait_enabled=0 ;;
esac

job_in_queue() {
  local job_id="$1"
  lpstat -W not-completed -o "$printer" 2>/dev/null | awk '{print $1}' | grep -qx "$job_id"
}

wait_for_job_completion() {
  local job_id="$1"
  [[ -z "$job_id" ]] && return 0
  while job_in_queue "$job_id"; do
    sleep 1
  done
}

wait_for_file_ready() {
  local f="$1"
  local tries=20
  local prev_size=-1 prev_mtime=-1
  while ((tries-- > 0)); do
    if [[ -r "$f" ]]; then
      local size mtime
      size=$(stat -f%z "$f" 2>/dev/null || echo -1)
      mtime=$(stat -f%m "$f" 2>/dev/null || echo -1)
      if [[ $size -gt 0 && $size -eq $prev_size && $mtime -eq $prev_mtime ]]; then
        return 0
      fi
      prev_size=$size
      prev_mtime=$mtime
    fi
    sleep 0.5
  done
  return 1
}

get_pages() {
  local f="$1"
  local pages=""
  pages=$(mdls -name kMDItemNumberOfPages -raw "$f" 2>/dev/null || true)
  if [[ "$pages" =~ ^[0-9]+$ ]]; then
    echo "$pages"
    return 0
  fi
  if command -v mdimport >/dev/null 2>&1; then
    mdimport "$f" >/dev/null 2>&1 || true
    pages=$(mdls -name kMDItemNumberOfPages -raw "$f" 2>/dev/null || true)
    if [[ "$pages" =~ ^[0-9]+$ ]]; then
      echo "$pages"
      return 0
    fi
  fi
  pages=$(sips -g pageCount "$f" 2>/dev/null | awk '/pageCount/ {print $2; exit}')
  if [[ "$pages" =~ ^[0-9]+$ ]]; then
    echo "$pages"
    return 0
  fi
  pages=$(/usr/bin/osascript - "$f" <<'APPLESCRIPT'
use framework "PDFKit"
use framework "Foundation"
use scripting additions
on run argv
  set p to item 1 of argv
  set theURL to current application's |NSURL|'s fileURLWithPath:p
  set thePDF to current application's PDFDocument's alloc()'s initWithURL:theURL
  if thePDF is not missing value then
    return (thePDF's pageCount() as integer) as string
  end if
end run
APPLESCRIPT
)
  pages=$(echo "$pages" | tr -d '[:space:]')
  if [[ "$pages" =~ ^[0-9]+$ ]]; then
    echo "$pages"
    return 0
  fi
  if command -v pdfinfo >/dev/null 2>&1; then
    pages=$(pdfinfo "$f" 2>/dev/null | awk '/^Pages:/ {print $2; exit}')
    if [[ "$pages" =~ ^[0-9]+$ ]]; then
      echo "$pages"
      return 0
    fi
  fi
  return 1
}

wait_for_file_ready "$file" || true
cache_dir="$HOME/Library/Caches/print2pages"
mkdir -p "$cache_dir"
work_file="$cache_dir/$(date +%s)-$$.pdf"
cp "$file" "$work_file"
trap 'rm -f "$work_file"' EXIT

pages=$(get_pages "$work_file" || true)
if [[ ! "$pages" =~ ^[0-9]+$ ]]; then
  echo "Could not determine PDF page count: $file" >&2
  exit 1
fi

start=1
while [[ $start -le $pages ]]; do
  end=$((start+1))
  if [[ $end -gt $pages ]]; then end=$pages; fi
  job_out=$(lp -d "$printer" -o page-ranges="${start}-${end}" "$work_file")
  echo "$job_out"
  job_id=$(echo "$job_out" | awk '{print $4}')
  if [[ $wait_enabled -eq 1 ]]; then
    wait_for_job_completion "$job_id"
  fi
  sleep "$delay"
  start=$((start+2))
done
EOF

chmod +x "$HOME/bin/print2pages-auto"
```

## Usage
Print a PDF with a 2‑second delay between each 2‑page chunk:

```zsh
"$HOME/bin/print2pages" "$HOME/Desktop/yourfile.pdf" 2
```

If you want to target a specific printer:

```zsh
"$HOME/bin/print2pages" "$HOME/Desktop/yourfile.pdf" 2 "Canon MF632C/634C"
```

### Optional: disable job‑completion waiting
By default, the script waits for each 2‑page job to finish before submitting the next.
If you want the old “submit then sleep” behavior, set `WAIT_FOR_JOB=0`:

```zsh
WAIT_FOR_JOB=0 "$HOME/bin/print2pages" "$HOME/Desktop/yourfile.pdf" 2
```

### Tip: drag‑and‑drop
In Terminal, you can drag a PDF from Finder into the command to insert its path.

## Automator (Split & Print)
Create an Automator **Print Plugin** and set its script to:

```zsh
"$HOME/bin/print2pages-auto" "$1" 2
```

Then use **Print → PDF → Split & Print** in any macOS app.

To disable job‑completion waiting in Automator:

```zsh
WAIT_FOR_JOB=0 "$HOME/bin/print2pages-auto" "$1" 2
```

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
- “Could not determine PDF page count”: the file may not be a PDF or is damaged.
- If jobs are still too fast, increase the delay (e.g. `10` or `12` seconds).
- If nothing prints, verify the printer name from `lpstat -p`.
- If Automator temp files fail to print, use `print2pages-auto` (not the direct script).

## What the delay means
With the default wait behavior, the delay happens **after a job finishes**, giving the printer a short settling break.
If you set `WAIT_FOR_JOB=0`, the delay is only the **pause between job submissions**.

## Why use $HOME instead of ~
On some keyboards, the tilde key inserts a **different** character (not ASCII `~`).
`$HOME` always works.