"""
Data analysis v2 - with fixed encoding and proper time overlap analysis.
"""
import os, struct, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(r"D:\VS Code-Projects\radar-3d-retrieval\data")

def analyze_time_overlap_with_tolerance():
    """Check time overlap with various tolerance windows."""
    print("=" * 70)
    print("TIME OVERLAP ANALYSIS (WITH TOLERANCE)")
    print("=" * 70)

    station_ts = {}
    for station in sorted(DATA_DIR.iterdir()):
        if not station.is_dir():
            continue
        ppi_dir = station / "PPI"
        if not ppi_dir.exists():
            continue

        timestamps = set()
        for elev_dir in ppi_dir.iterdir():
            if elev_dir.is_dir():
                for f in elev_dir.iterdir():
                    if f.name == "ProductIndex":
                        continue
                    parts = f.name.split("_")
                    if len(parts) >= 2:
                        timestamps.add(parts[1])

        if timestamps:
            station_ts[station.name] = sorted(timestamps)

    for sname, tss in station_ts.items():
        print(f"\n{sname}: {len(tss)} timestamps")
        dts = [datetime.strptime(ts, "%Y%m%d%H%M%SZ") for ts in tss]
        print(f"  First: {dts[0]}, Last: {dts[-1]}")
        if len(dts) >= 2:
            intervals = [(dts[i+1] - dts[i]).total_seconds()/60 for i in range(len(dts)-1)]
            print(f"  Scan interval: {min(intervals):.1f} - {max(intervals):.1f} min")

    # Match with tolerance
    if len(station_ts) >= 2:
        stations = sorted(station_ts.keys())
        for tolerance_min in [1, 2, 3, 4, 5, 6, 7, 10]:
            s1, s2 = stations[0], stations[1]
            dts1 = [(datetime.strptime(ts, "%Y%m%d%H%M%SZ"), ts) for ts in station_ts[s1]]
            dts2 = [(datetime.strptime(ts, "%Y%m%d%H%M%SZ"), ts) for ts in station_ts[s2]]

            matches = []
            for dt1, ts1 in dts1:
                closest_dt2, closest_ts2 = min(dts2, key=lambda x: abs(x[0] - dt1))
                diff_min = abs((dt1 - closest_dt2).total_seconds()) / 60
                if diff_min <= tolerance_min:
                    matches.append((dt1, ts1, closest_dt2, closest_ts2, diff_min))

            # Remove duplicate matches from s2 side
            used_s2 = set()
            unique_matches = []
            for m in matches:
                if m[3] not in used_s2:
                    used_s2.add(m[3])
                    unique_matches.append(m)

            if unique_matches:
                print(f"\n  Tolerance +/-{tolerance_min} min: {len(unique_matches)} matching pairs")
                for dt1, ts1, dt2, ts2, diff in unique_matches:
                    print(f"    Z9317={dt1.strftime('%H:%M:%S')} <-> Z9543={dt2.strftime('%H:%M:%S')} (diff={diff:.1f} min)")
            else:
                print(f"  Tolerance +/-{tolerance_min} min: 0 matches")

        # Also show all timestamps side by side for visual comparison
        print(f"\n  Side-by-side comparison:")
        print(f"  {'Z9317':<20s}  {'Z9543':<20s}")
        print(f"  {'-'*20}  {'-'*20}")
        max_len = max(len(station_ts[stations[0]]), len(station_ts[stations[1]]))
        dts1 = [datetime.strptime(ts, "%Y%m%d%H%M%SZ") for ts in station_ts[stations[0]]]
        dts2 = [datetime.strptime(ts, "%Y%m%d%H%M%SZ") for ts in station_ts[stations[1]]]
        for i in range(max_len):
            s1_str = dts1[i].strftime('%H:%M:%S') if i < len(dts1) else ""
            s2_str = dts2[i].strftime('%H:%M:%S') if i < len(dts2) else ""
            print(f"  {s1_str:<20s}  {s2_str:<20s}")

        print(f"\n  == ANALYSIS ==")
        print(f"  The two radars have DIFFERENT scanning schedules (~5 min vs ~6 min intervals)")
        print(f"  This means exact time matches are impossible.")
        print(f"  For dual-Doppler analysis, the acceptable time window is typically 5-10 minutes.")
        print(f"  Given 6-min scan intervals, a +/-5 min tolerance should work.")

        # Find the best matching pairs
        print(f"\n  Best match for each Z9317 volume (within 6 min):")
        dts1 = [(datetime.strptime(ts, "%Y%m%d%H%M%SZ"), ts) for ts in station_ts[s1]]
        dts2 = [(datetime.strptime(ts, "%Y%m%d%H%M%SZ"), ts) for ts in station_ts[s2]]
        for dt1, ts1 in dts1:
            closest_dt2, closest_ts2 = min(dts2, key=lambda x: abs(x[0] - dt1))
            diff_min = abs((dt1 - closest_dt2).total_seconds()) / 60
            marker = "OK" if diff_min <= 6 else "MARGINAL" if diff_min <= 10 else "POOR"
            print(f"    Z9317 {dt1.strftime('%H:%M:%S')} -> Z9543 {closest_dt2.strftime('%H:%M:%S')} (diff={diff_min:.1f} min) [{marker}]")

    return station_ts


def analyze_file_structure_deep():
    """Deep analysis of file binary structure."""
    print("\n" + "=" * 70)
    print("DEEP FILE FORMAT ANALYSIS")
    print("=" * 70)

    for station in sorted(DATA_DIR.iterdir()):
        if not station.is_dir():
            continue

        # Compare 019 vs 19 directories (different sizes)
        for elev_code in ["019", "19"]:
            elev_dir = station / "PPI" / elev_code
            if not elev_dir.exists():
                continue

            files = sorted([f for f in elev_dir.iterdir() if f.name != "ProductIndex"])
            if not files:
                continue

            sample = files[0]
            fsize = sample.stat().st_size
            print(f"\n{station.name}/PPI/{elev_code}/{sample.name}")
            print(f"  Size: {fsize} bytes")

            with open(sample, "rb") as f:
                data = f.read()

            # Read potential header
            # Looking for structure: station_id, VCP, elevation code, azimuth info
            # In CINRAD Level-II product format:

            # Try to extract header fields
            # Check if the data starts with a header by looking for typical patterns

            # For CINRAD WRS (Weather Radar Software) format:
            # Each product file has a specific header structure
            # Let's look at what's readable in the first 1024 bytes
            print(f"  Readable strings in header (first 512 bytes):")
            for line in data[:512].split(b'\x00'):
                line = line.strip()
                if len(line) > 2:
                    try:
                        decoded = line.decode('ascii', errors='replace')
                        if any(c.isalnum() for c in decoded):
                            print(f"    '{decoded}'")
                    except:
                        pass

            # Check for known radar-specific patterns
            # Common header sizes in CINRAD: 64, 128, 256, 1024, 2048, 4096 bytes
            # Look for pattern where header ends

            # The ProductIndex shows: station_id + VCP + elevation + layer info
            # Possible structure (from ProductIndex):
            # Each entry has: station name, VCP string, timestamps?, elevation code, layer number

            # Let's calculate if data dimensions can be inferred
            if fsize > 1024:
                # Try common radar grid sizes
                # CINRAD typically has 360 or 720 azimuths and 460 or 920 range gates
                for az in [360, 720, 1440]:
                    for rg in [230, 460, 920, 1000]:
                        expected = 1024 + az * rg  # header + data
                        if abs(fsize - expected) < 2000:
                            print(f"  Possible grid: {az} azimuths x {rg} range gates (+~1024 header) = {expected}")
                        expected2 = 2048 + az * rg
                        if abs(fsize - expected2) < 2000:
                            print(f"  Possible grid: {az} azimuths x {rg} range gates (+~2048 header) = {expected2}")
                        expected3 = 128 + az * rg * 2  # 2 bytes per pixel
                        if abs(fsize - expected3) < 2000:
                            print(f"  Possible grid: {az} azimuths x {rg} range gates x 2bytes (+~128 header) = {expected3}")

def analyze_pixel_data():
    """Try to decode the actual radar pixel data."""
    print("\n" + "=" * 70)
    print("PIXEL DATA ANALYSIS")
    print("=" * 70)

    for station in sorted(DATA_DIR.iterdir()):
        if not station.is_dir():
            continue

        # Compare same elevation from 019 (large) and 19 (small) directories
        for elev_code in ["019", "19"]:
            elev_dir = station / "PPI" / elev_code
            if not elev_dir.exists():
                continue

            files = sorted([f for f in elev_dir.iterdir() if f.name != "ProductIndex"])
            if not files:
                continue

            sample = files[0]
            with open(sample, "rb") as f:
                data = f.read()

            fsize = len(data)

            # The header size might be identifiable by looking for the end
            # of ASCII text and start of binary data
            # Let's look for the transition point

            # Read as shorts (2-byte signed) in various regions
            # For radar velocity, expect values around -64 to +64 m/s
            # scaled by some factor (typically 100 or 10)

            print(f"\n{station.name}/PPI/{elev_code}/layer_01")
            print(f"  File size: {fsize} bytes")

            # Try interpreting as 1-byte values (biased/offset data)
            if fsize > 2000:
                # CINRAD WRS format: header may be 1024, 2048, or variable
                for header_size in [0, 128, 256, 512, 1024, 2048, 4096]:
                    if header_size >= fsize:
                        continue

                    body = data[header_size:]
                    body_len = len(body)

                    # Try as unsigned bytes
                    hist = defaultdict(int)
                    for b in body[:100000]:
                        hist[b] += 1

                    # Sort by frequency
                    sorted_vals = sorted(hist.items(), key=lambda x: -x[1])

                    # Look for characteristic patterns:
                    # - Many zeros (no echo)
                    # - Most values in a specific range
                    # - Avoidance of certain values (e.g., missing/bad data codes)

                    zero_count = hist.get(0, 0)
                    total_pct = zero_count / min(100000, body_len) * 100

                    # Check values in typical radar range
                    # For dBZ: 0-70 typically, no-echo = 0 or special code
                    # For velocity: bias-shifted from -Vmax to +Vmax

                    typical_range = sum(1 for b in body[:50000] if 1 <= b <= 200)
                    total_range_pct = typical_range / min(50000, body_len) * 100

                    if total_pct > 30 and total_range_pct > 20:
                        print(f"    Header={header_size}: {zero_count} zeros ({total_pct:.1f}%), "
                              f"typical range 1-200: {total_range_pct:.1f}%")
                        print(f"    Top byte values: {sorted_vals[:8]}")

                        # Check if values could represent dBZ (reflectivity)
                        # dBZ typically: 5-75, with 0 = no echo
                        dbz_range = sum(1 for b in body[:50000] if 5 <= b <= 75)
                        # Check if values could represent velocity
                        # Velocity typically: centered around some bias value
                        vel_center = max(hist, key=hist.get) if hist else 0
                        vel_range = sum(1 for b in body[:50000] if abs(int(b) - vel_center) < 30)

                        if dbz_range > 0.4 * min(50000, body_len):
                            print(f"      -> Likely REFLECTIVITY data (dBZ range)")
                        elif vel_range > 0.3 * min(50000, body_len):
                            print(f"      -> Possibly VELOCITY data (centered near byte value {vel_center})")


def product_index_analysis():
    """Deep analysis of ProductIndex files."""
    print("\n" + "=" * 70)
    print("PRODUCT INDEX ANALYSIS")
    print("=" * 70)

    for station in sorted(DATA_DIR.iterdir()):
        if not station.is_dir():
            continue

        ppi_019 = station / "PPI" / "019"
        if ppi_019.exists():
            idx_file = ppi_019 / "ProductIndex"
            if idx_file.exists():
                print(f"\n{station.name}/PPI/019/ProductIndex:")
                print(f"  Size: {idx_file.stat().st_size} bytes")

                with open(idx_file, "rb") as f:
                    data = f.read()

                # Each entry seems to have a fixed structure
                # The ProductIndex lists all files within this product/elevation
                # Let's find the record size

                # Read all printable segments
                print(f"  Content segments:")
                entries = data.split(b'Z9317' if station.name == 'Z9317' else b'Z9543')
                print(f"  Number of entries: {len(entries) - 1}")

                # Look at one complete entry
                if len(entries) > 1:
                    entry = station.name.encode() + entries[1]
                    print(f"  First entry size: {len(entry)} bytes")

                    # Show hex dump of one entry
                    print(f"  First entry hex (first 256 bytes):")
                    for i in range(0, min(256, len(entry)), 16):
                        hex_str = " ".join(f"{b:02x}" for b in entry[i:i+16])
                        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in entry[i:i+16])
                        print(f"  {i:04x}: {hex_str:<48s}  {ascii_str}")

                    # Parse the record structure
                    # From hex, identify:
                    # - Station name
                    # - VCP type (VCP11D)
                    # - Timestamps
                    # - Elevation code
                    # - Product type
                    # - Layer number
                    # - Data format info
                    print(f"\n  Record structure analysis:")
                    # Station name at offset ~0
                    stn = entry[0:5].decode('ascii', errors='ignore')
                    print(f"    Station: {stn}")

                    # VCP at offset after station
                    for i in range(len(entry)):
                        if entry[i:i+6] == b'VCP11D':
                            print(f"    VCP type at offset {i}: VCP11D")
                            break

                    # Elevation code
                    for elev in [b'019', b'020', b'026', b'027', b'158', b'160', b'161', b'162']:
                        idx = entry.find(elev)
                        if idx >= 0:
                            print(f"    Elevation code at offset {idx}: {elev.decode()}")
                            break

                    # File name
                    for i in range(len(entry)-30):
                        if entry[i:i+1] == station.name.encode()[0:1]:
                            try:
                                chunk = entry[i:i+60]
                                decoded = chunk.decode('ascii', errors='ignore')
                                if '_' in decoded and len(decoded.split('_')) >= 4:
                                    print(f"    Filename at offset {i}: {decoded.strip(chr(0)).strip()}")
                                    break
                            except:
                                pass

                # Calculate record size
                if len(entries) > 2:
                    e1 = station.name.encode() + entries[1]
                    e2 = station.name.encode() + entries[2]
                    rec_size = len(e1)
                    print(f"    Record size: {rec_size} bytes")


def main():
    print("=" * 70)
    print("RADAR DATA ANALYSIS FOR PyDDA/3DVAR WIND RETRIEVAL")
    print("=" * 70)

    analyze_time_overlap_with_tolerance()
    analyze_file_structure_deep()
    product_index_analysis()
    analyze_pixel_data()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("""
DATA IDENTIFIED:
  - Source: CINRAD WRS (Weather Radar Software) products
  - VCP: VCP11D (Volume Coverage Pattern 11D)
  - Resolution: ~6 min per volume scan
  - PPI layers: 01-06 per elevation angle
  - Content: Mixture of reflectivity and radial velocity

FOR PyDDA / 3DVAR WIND RETRIEVAL:
  1. Need to identify which PPI layers contain radial velocity
     - In homogeneous CINRAD products:
       odd layers (01, 03, 05) = Reflectivity
       even layers (02, 04, 06) = Radial Velocity / Spectrum Width

  2. From the ProductIndex, we see 6 layers at each elevation:
     - PPI_01_019 through PPI_06_019 at 0.5 degrees
     - These likely correspond to:
       01: Reflectivity at 0.5 deg
       02: Radial Velocity at 0.5 deg
       03: Spectrum Width at 0.5 deg
       04: Reflectivity at 1.5 deg
       05: Radial Velocity at 1.5 deg
       06: Spectrum Width at 1.5 deg

  3. The two file sizes (353KB vs 100KB) suggest:
     - 353KB: Full resolution product data (019, 020, 158, 160, 161, 162 dirs)
     - 100KB: Compressed/lower-res product data (19, 20, 26, 27 dirs)
     These may be duplicates at different post-processing stages.

  4. For wind retrieval:
     - Use PPI layers 02, 05 (= radial velocity)
     - From all available elevation angles
     - Match time-adjacent volumes between both radars

NEXT STEPS:
  1. Install required packages:
     pip install pyart pydda cartopy numpy scipy matplotlib

  2. Read binary files using PyART's radar file readers

  3. Identify velocity layers and extract radial velocity fields

  4. Set up dual-Doppler geometry using radar coordinates

  5. Run 3DVAR wind retrieval
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
