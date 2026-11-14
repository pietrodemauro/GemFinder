#!/usr/bin/env bash
# GemFinder Quick Toggle Script for Global Keybinding (e.g. Super+Space or Alt+Space)

set -euo pipefail

# Check if gemfinder-gui is running
if pgrep -x "gemfinder-gui" > /dev/null; then
    # Try focusing via wmctrl or hyprctl / swaymsg if available
    if command -v hyprctl &>/dev/null; then
        hyprctl dispatch focuswindow "initialTitle:GemFinder" || pkill -x gemfinder-gui
    elif command -v swaymsg &>/dev/null; then
        swaymsg '[app_id="gemfinder-gui"] focus' || pkill -x gemfinder-gui
    elif command -v wmctrl &>/dev/null; then
        wmctrl -x -a "gemfinder-gui.gemfinder-gui" || pkill -x gemfinder-gui
    else
        pkill -x gemfinder-gui
    fi
else
    # Launch GUI in background
    gemfinder-gui &
fi
