#!/usr/bin/env python3
"""
ros2_qos_probe.py

Measure pub-sub latency and message loss under configurable rate/size. If rclpy is available,
uses ROS 2 intra-process pub/sub; otherwise falls back to a local UDP loopback probe.

Usage (ROS 2):
  python ros2_qos_probe.py --mode ros2 --hz 50 --size 1024 --duration-sec 60 --out logs/ros2_qos.csv
Usage (UDP fallback):
  python ros2_qos_probe.py --mode udp --hz 50 --size 1024 --duration-sec 60 --out logs/ros2_qos.csv

CSV: ts_iso,seq,tx_ns,rx_ns,latency_ms,status
"""
import argparse
import os
import socket
import sys
import time
from datetime import datetime


def write_header(f):
    f.write("ts_iso,seq,tx_ns,rx_ns,latency_ms,status\n")


def udp_probe(hz: float, size: int, duration: int, out: str):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    period = 1.0 / max(1e-6, hz)
    deadline = time.perf_counter() + max(1, duration)
    seq = 0
    # UDP loopback
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.bind(("127.0.0.1", 0))
    rx_port = sock_rx.getsockname()[1]
    sock_rx.settimeout(0.1)
    payload = bytearray(size)
    with open(out, "w") as f:
        write_header(f)
        try:
            while time.perf_counter() < deadline:
                t0 = time.perf_counter()
                tx_ns = time.time_ns()
                # write seq and tx_ns at the beginning if enough room
                header = seq.to_bytes(8, "big") + tx_ns.to_bytes(8, "big")
                payload[: len(header)] = header
                sock_tx.sendto(payload, ("127.0.0.1", rx_port))
                # receive
                try:
                    data, _ = sock_rx.recvfrom(65536)
                    rx_ns = time.time_ns()
                    if len(data) >= 16:
                        rx_seq = int.from_bytes(data[:8], "big")
                        rx_tx_ns = int.from_bytes(data[8:16], "big")
                    else:
                        rx_seq = seq
                        rx_tx_ns = tx_ns
                    lat_ms = (rx_ns - rx_tx_ns) / 1e6
                    ts = datetime.utcnow().isoformat() + "Z"
                    f.write(f"{ts},{rx_seq},{rx_tx_ns},{rx_ns},{lat_ms:.3f},ok\n")
                except socket.timeout:
                    ts = datetime.utcnow().isoformat() + "Z"
                    f.write(f"{ts},{seq},{tx_ns},,,-\n")
                seq += 1
                # sleep until next
                t0 += period
                dt = max(0.0, t0 - time.perf_counter())
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            pass
    print("[i] Wrote:", out)


def ros2_probe(hz: float, size: int, duration: int, out: str):
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import ByteMultiArray  # type: ignore
    except Exception:
        print("[w] rclpy not available; falling back to UDP probe.")
        return udp_probe(hz, size, duration, out)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    period = 1.0 / max(1e-6, hz)

    class QosNode(Node):
        def __init__(self):
            super().__init__("qos_probe")
            self.pub = self.create_publisher(ByteMultiArray, "qos_topic", 10)
            self.sub = self.create_subscription(
                ByteMultiArray, "qos_topic", self.on_msg, 10
            )
            self.timer = self.create_timer(period, self.on_tick)
            self.seq = 0
            self.size = max(16, size)
            self.deadline = self.get_clock().now().nanoseconds + int(duration * 1e9)
            self.f = open(out, "w")
            write_header(self.f)

        def on_tick(self):
            now_ns = self.get_clock().now().nanoseconds
            if now_ns >= self.deadline:
                self.f.close()
                rclpy.shutdown()
                return
            msg = ByteMultiArray()
            buf = bytearray(self.size)
            hdr = self.seq.to_bytes(8, "big") + now_ns.to_bytes(8, "big")
            buf[: len(hdr)] = hdr
            msg.data = list(buf)
            self.pub.publish(msg)
            self.seq += 1

        def on_msg(self, msg: "ByteMultiArray"):
            now_ns = self.get_clock().now().nanoseconds
            data = bytes(msg.data)
            if len(data) >= 16:
                seq = int.from_bytes(data[:8], "big")
                tx_ns = int.from_bytes(data[8:16], "big")
            else:
                seq = -1
                tx_ns = now_ns
            lat_ms = (now_ns - tx_ns) / 1e6
            ts = datetime.utcnow().isoformat() + "Z"
            self.f.write(f"{ts},{seq},{tx_ns},{now_ns},{lat_ms:.3f},ok\n")

    rclpy.init()
    node = QosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.f.close()
        except Exception:
            pass
    print("[i] Wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ros2", "udp"], default="ros2")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--duration-sec", type=int, default=60)
    ap.add_argument("--out", default="logs/ros2_qos.csv")
    args = ap.parse_args()
    if args.mode == "ros2":
        ros2_probe(args.hz, args.size, args.duration_sec, args.out)
    else:
        udp_probe(args.hz, args.size, args.duration_sec, args.out)


if __name__ == "__main__":
    main()
