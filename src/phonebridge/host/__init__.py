"""The Mac phonebridge runs on (DEV-04, REC-05): Ian's daily MacBook, "on when it's on" (DECISIONS Q1).

power      caffeinate -i -w <pid> only while a task runs, and the busy marker deploys check
watch      sleep/wake, USB attach/detach, thermal/battery -> park the bridge, re-check, resume
heartbeat  a beat to the VM every 60 s while awake (the VM's dead-man is authhub.deadman); log caps
preflight  the probe table and operator attestations (DEV-01/02/08, NET-01)
recover    L3 device and L4 host resets, one attempt each, then Ian (REC-04)
"""
