
#!/usr/bin/env python3
"""
Flock of Birds POSITION/ANGLES reader for Linux + pyserial.
"""

import argparse
import sys
import time
from dataclasses import dataclass

try:
    import serial
except Exception:
    print("[ERROR] pyserial이 필요합니다. pip install pyserial")
    raise


POINT_CMD = 0x42
POS_ANGLES_CMD = 0x59
RECORD_LEN = 12


@dataclass
class PoseAngles:
    x_in: float
    y_in: float
    z_in: float
    x_cm: float
    y_cm: float
    z_cm: float
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    raw_words: tuple


def hexdump(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def open_serial(port: str, baud: int, timeout: float):
    return serial.Serial(
        port=port,
        baudrate=baud,
        timeout=timeout,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )


def decode_fob_word(lsb_byte: int, msb_byte: int) -> int:
    word = ((msb_byte & 0x7F) << 9) | ((lsb_byte & 0x7F) << 2)
    if word >= 0x8000:
        word -= 0x10000
    return word


def position_to_inches(raw: int, max_range_in: float) -> float:
    return (raw * max_range_in) / 32768.0


def angle_to_degrees(raw: int) -> float:
    return (raw * 180.0) / 32768.0


def parse_position_angles_record(record: bytes, max_range_in: float) -> PoseAngles:
    if len(record) != RECORD_LEN:
        raise ValueError(f"Expected {RECORD_LEN} bytes, got {len(record)}")

    words = []
    for i in range(0, RECORD_LEN, 2):
        lsb_b = record[i]
        msb_b = record[i + 1]
        words.append(decode_fob_word(lsb_b, msb_b))

    x_raw, y_raw, z_raw, zang_raw, yang_raw, xang_raw = words

    x_in = position_to_inches(x_raw, max_range_in)
    y_in = position_to_inches(y_raw, max_range_in)
    z_in = position_to_inches(z_raw, max_range_in)

    x_cm = x_in * 2.54
    y_cm = y_in * 2.54
    z_cm = z_in * 2.54

    azimuth_deg = angle_to_degrees(zang_raw)
    elevation_deg = angle_to_degrees(yang_raw)
    roll_deg = angle_to_degrees(xang_raw)

    return PoseAngles(
        x_in=x_in,
        y_in=y_in,
        z_in=z_in,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        roll_deg=roll_deg,
        raw_words=(x_raw, y_raw, z_raw, zang_raw, yang_raw, xang_raw),
    )


def read_exact(ser, n: int, deadline_s: float) -> bytes:
    out = bytearray()
    t0 = time.time()
    while len(out) < n:
        if time.time() - t0 > deadline_s:
            break
        chunk = ser.read(n - len(out))
        if chunk:
            out.extend(chunk)
    return bytes(out)


def phasing_ok(record: bytes) -> bool:
    if len(record) != RECORD_LEN:
        return False
    if (record[0] & 0x80) == 0:
        return False
    for b in record[1:]:
        if (b & 0x80) != 0:
            return False
    return True


def read_one_position_angles_record(ser, timeout_s: float) -> bytes:
    ser.write(bytes([POINT_CMD]))
    ser.flush()

    record = read_exact(ser, RECORD_LEN, timeout_s)
    if phasing_ok(record):
        return record

    buf = bytearray(record)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        b = ser.read(1)
        if b:
            buf.extend(b)

        while len(buf) >= RECORD_LEN:
            candidate = bytes(buf[:RECORD_LEN])
            if phasing_ok(candidate):
                return candidate
            del buf[0]

    raise TimeoutError("12바이트 POSITION/ANGLES 레코드를 phasing 기준으로 읽지 못했습니다.")


def main():
    parser = argparse.ArgumentParser(description="Flock of Birds POSITION/ANGLES reader")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial port")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate")
    parser.add_argument("--timeout", type=float, default=0.2, help="serial read timeout")
    parser.add_argument("--count", type=int, default=0, help="number of samples, 0 means infinite")
    parser.add_argument("--period", type=float, default=0.1, help="seconds between samples")
    parser.add_argument("--range-in", type=float, default=36.0,
                        help="position full-scale range in inches (common: 36, 72, 144)")
    parser.add_argument("--set-pos-angles", action="store_true",
                        help="send 0x59 ('Y') once at startup")
    parser.add_argument("--raw", action="store_true",
                        help="also print raw bytes and decoded raw words")
    args = parser.parse_args()

    try:
        ser = open_serial(args.port, args.baud, args.timeout)
    except Exception as e:
        print(f"[ERROR] 포트를 열 수 없습니다: {e}")
        sys.exit(1)

    print(f"[INFO] opened port={args.port}, baud={args.baud}, timeout={args.timeout}")
    print(f"[INFO] assumed format=POSITION/ANGLES (12 bytes), range={args.range_in} in")

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        if args.set_pos_angles:
            ser.write(bytes([POS_ANGLES_CMD]))
            ser.flush()
            time.sleep(0.05)
            ser.reset_input_buffer()

        sample_idx = 0
        while True:
            if args.count > 0 and sample_idx >= args.count:
                break

            sample_idx += 1
            try:
                record = read_one_position_angles_record(ser, timeout_s=max(0.5, args.timeout * 4))
                pose = parse_position_angles_record(record, max_range_in=args.range_in)

                print(
                    f"[{sample_idx:04d}] "
                    f"X={pose.x_cm:8.3f} cm  "
                    f"Y={pose.y_cm:8.3f} cm  "
                    f"Z={pose.z_cm:8.3f} cm   |   "
                    f"Az={pose.azimuth_deg:7.3f} deg  "
                    f"El={pose.elevation_deg:7.3f} deg  "
                    f"Roll={pose.roll_deg:7.3f} deg"
                )

                if args.raw:
                    print(f"        raw_bytes: {hexdump(record)}")
                    print(f"        raw_words: {pose.raw_words}")

            except Exception as e:
                print(f"[WARN] sample {sample_idx}: {e}")

            time.sleep(args.period)

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단으로 종료합니다.")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("[INFO] serial closed")


if __name__ == "__main__":
    main()
