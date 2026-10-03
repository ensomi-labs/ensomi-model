#!/bin/sh
# Manage the launchd user agent that runs dump_collector.py once a day on bings-mac.
#
#   scripts/player_data/launchd.sh install [HOUR MINUTE]   write the plist, load it (default 21:15 local)
#   scripts/player_data/launchd.sh status                  launchctl list / print for the label
#   scripts/player_data/launchd.sh kick                    start one run now
#   scripts/player_data/launchd.sh uninstall               unload and delete the plist
#   scripts/player_data/launchd.sh print-plist [HOUR MINUTE]
#
# The plist holds absolute paths, so it is generated here at install time and written only to
# ~/Library/LaunchAgents, which is not synced. Logs go to artifacts/player-data-dumps/log/.
set -eu

LABEL=com.ensomi.player-dump-collector
REPO=$(cd "$(dirname "$0")/../.." && pwd -P)
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$REPO/artifacts/player-data-dumps/log"
DOMAIN="gui/$(id -u)"
CMD=${1:-}
HOUR=${2:-21}
MINUTE=${3:-15}

plist() {
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$REPO/.venv/bin/python</string>
        <string>$REPO/scripts/player_data/dump_collector.py</string>
        <string>--job-id</string>
        <string>launchd</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$REPO</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>$HOUR</integer>
        <key>Minute</key>
        <integer>$MINUTE</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>ProcessType</key>
    <string>Background</string>
    <key>Nice</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>$LOGDIR/launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>$LOGDIR/launchd.err.log</string>
</dict>
</plist>
EOF
}

require_mac() {
  [ "$(uname -s)" = Darwin ] || { echo "launchd agents exist only on macOS (bings-mac)" >&2; exit 1; }
}

case "$CMD" in
  install)
    require_mac
    [ -x "$REPO/.venv/bin/python" ] || { echo "missing $REPO/.venv/bin/python" >&2; exit 1; }
    mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"
    plist > "$PLIST.tmp"
    plutil -lint "$PLIST.tmp" >/dev/null
    mv "$PLIST.tmp" "$PLIST"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    if launchctl bootstrap "$DOMAIN" "$PLIST"; then
      echo "loaded with: launchctl bootstrap $DOMAIN $PLIST"
    else
      echo "launchctl bootstrap failed; trying the legacy launchctl load -w" >&2
      launchctl load -w "$PLIST"
      echo "loaded with: launchctl load -w $PLIST"
    fi
    launchctl list | grep -F "$LABEL" || { echo "not listed after loading" >&2; exit 1; }
    echo "runs daily at $(printf '%02d:%02d' "$HOUR" "$MINUTE") local time; logs in $LOGDIR"
    ;;
  status)
    require_mac
    launchctl list | grep -F "$LABEL" || { echo "$LABEL is not loaded"; exit 1; }
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null \
      | grep -E '^\s*(state|program|last exit code|runs|path) =' || true
    ;;
  kick)
    require_mac
    launchctl kickstart "$DOMAIN/$LABEL" 2>/dev/null || launchctl start "$LABEL"
    echo "started $LABEL; see $LOGDIR/launchd.out.log and $LOGDIR/runs.jsonl"
    ;;
  uninstall)
    require_mac
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || launchctl unload -w "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    if launchctl list | grep -qF "$LABEL"; then echo "still listed" >&2; exit 1; fi
    echo "removed $LABEL ($PLIST deleted; collected data untouched)"
    ;;
  print-plist)
    plist
    ;;
  *)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
