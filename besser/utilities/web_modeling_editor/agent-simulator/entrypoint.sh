#!/bin/bash
set -e

mkdir -p /tmp/sessions
chown root:root /tmp/sessions
chmod 0711 /tmp/sessions

exec python agent_simulator_api.py
