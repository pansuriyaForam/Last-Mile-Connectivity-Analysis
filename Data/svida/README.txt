Hyd SVIDA Red Line GTFS Feed
Generated: 2026-04-07

Scope
- Red Line feeder services only
- Miyapur Routes 01-08
- LB Nagar Routes 01-03

Source hierarchy
1. Uploaded HMRL metro stops.txt for exact metro pickup-arm coordinates (MYP_ENT03, LBN_ENT05)
2. Uploaded PDF "GTFS Data Generation for Feeder Services.pdf" for most feeder stop coordinates and GTFS design assumptions
3. Official HMRL last-mile connectivity page for current route names, timings, headways, fares, and service days

Important assumptions
- This is a project-grade static GTFS feed, not an official operator release.
- Exact path geometry was not published; shapes follow stop sequence straight lines.
- Several locality coordinates were approximated from public map-style web results:
  S_MXR (Miyapur X Road), S_LBR (LB Nagar Ring Road), S_PAN (Panama X Road), S_OMK (Omkar Nagar), S_HAS (Hastinapuram).
- Where the official page listed only origin and destination, no intermediate stops were invented.
- Frequencies are encoded using morning/evening service windows from the official HMRL page.
- stop_times are estimated from inter-stop distance using an average feeder operating speed model.

Suggested use
- Merge this feed with the metro GTFS in your notebook.
- Keep coordinate_sources.csv with the feed so you can defend which stops are exact vs approximate.
