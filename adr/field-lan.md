# ADR: Concurrent STA+AP with Policy‑Routed WireGuard for Portable “FieldLAN”

- ADR ID: ADR-STA-AP-WG-PR-001
- Date: 2025-08-25
- Status: Accepted
- Decision owner: AI Assistant
- Stakeholders: Networking/Platform, Demo team
- Context: Laptop “pluton” with Intel AX211 (iwlwifi) must:
  - Stay connected to upstream Wi‑Fi (client/STA) for Internet.
  - Provide a local hotspot (AP) for field devices.
  - Optionally route only hotspot clients over a WireGuard tunnel (home site) while host traffic uses direct WAN.
  - Solution must be resilient for a live demo and respect ISP/workplace policies.

## Decision

Implement single‑radio concurrency (STA + AP) on the AX211 using a dedicated virtual AP interface and NetworkManager, and add policy routing so only AP clients traverse WireGuard.

- Wireless concurrency
  - Use wlp0s20f3 as STA (connection “HomeWAN”).
  - Create virtual AP interface ap0 (type AP) on the same PHY (phy#0).
  - Bind AP profile “FieldLAN” to ap0, IPv4 method “shared” (NAT, DHCP via NM).
  - Constrain both STA and AP to the same 2.4 GHz channel (ch 1) to satisfy hardware limit “#channels <= 1”.

- VPN and routing
  - WireGuard interface wg0 connects to “Home Base.”
  - Policy routing: only 10.42.0.0/24 (FieldLAN clients) are marked and routed via routing table wgonly → default via wg0; host (laptop) keeps default route via HomeWAN.

- Boot persistence and activation order
  - systemd oneshot service creates ap0 (type AP) before NetworkManager starts.
  - NetworkManager dispatcher hook brings up FieldLAN after HomeWAN connects.

- Security and compliance
  - WPA2-PSK passphrase length ≥ 8 (corrected from initial 5‑char PSK).
  - AP is private for demo devices only.
  - ISP CGU reviewed at high level; no explicit VPN ban found. Treat use as personal/non‑commercial; avoid resale/public offering; confirm with ISP if ongoing.

## Alternatives considered

- Dedicated second Wi‑Fi NIC (USB dongle)
  - Pros: simpler, no single‑radio airtime contention; independent channels.
  - Cons: extra hardware; not always available.
- hostapd + manual DHCP/NAT
  - Pros: granular control.
  - Cons: more moving parts; NetworkManager integration work.
- Route everything (host + AP clients) via wg0
  - Pros: simpler routing.
  - Cons: demo host loses direct WAN path; less flexible.

## Rationale

- AX211 supports concurrent #{managed, AP} with “#channels <= 1” (confirmed by iw list).
- NetworkManager 1.54 supports multi‑connect and IPv4 sharing; however, it tends to reuse the same netdev unless the AP runs on a separate VIF. Creating ap0 solves that.
- Policy routing isolates demo traffic (AP clients) to the VPN while keeping the host on low‑latency local WAN.

## Architecture overview

- Interfaces
  - wlp0s20f3 (managed): HomeWAN, DHCP from upstream, default route (metric 100).
  - ap0 (AP): FieldLAN, 10.42.0.1/24 handed out via NM’s embedded dnsmasq, NAT to uplink.
  - wg0 (WireGuard): 10.81.166.4/24 to Home Base.

- Routing
  - nftables sets mark 0x1 on IPv4 traffic originating from 10.42.0.0/24 (and locally from 10.42.0.1).
  - ip rule fwmark 0x1 → table wgonly; table wgonly has default dev wg0.
  - PostUp/PreDown hooks in wg0 manage nft and ip rules.

## Implementation highlights

- NetworkManager profiles
  - HomeWAN: autoconnect yes; band bg; BSSID pinned; connection.multi-connect multiple; route-metric 100.
  - FieldLAN: interface ap0; mode ap; band bg; channel 1; ipv4.method shared; connection.autoconnect no; connection.multi-connect multiple; PSK length ≥ 8.

- System services
  - ap0-create.service (oneshot) creates ap0 type __ap before NM.
  - NM dispatcher script starts FieldLAN when HomeWAN goes up.

- WireGuard
  - wg-quick manages wg0; PostUp sets nft rules, ip rule, and route for table wgonly; PreDown removes them.

## Consequences

- Benefits
  - Portable field hotspot with encrypted backhaul via home site.
  - Host keeps direct WAN for control/monitoring while clients use VPN.
  - Minimal extra hardware; leverages existing AX211.

- Trade-offs / Risks
  - Single‑radio airtime sharing reduces aggregate throughput; uplink limited by home upload.
  - AP must share STA channel; STA roaming/channel changes can disrupt AP.
  - NM device discovery quirks: ap0 may need recreation after driver reloads; mitigated with systemd service.
  - dnsmasq under NM can occasionally hiccup; restart FieldLAN to recover.

## Operations (runbook)

- Start order (manual)
  - nmcli con up HomeWAN
  - nmcli con up FieldLAN ifname ap0
- Verify
  - iw dev → two interfaces: wlp0s20f3 (managed), ap0 (AP), same channel.
  - ip -brief addr → wlp0s20f3 has 192.168.1.x; ap0 has 10.42.0.1/24.
  - ip route → default via wlp0s20f3; 10.42.0.0/24 via ap0; wgonly table default via wg0.
  - From AP client, curl ifconfig.me shows VPN egress; from host, WAN egress.
- Quick recovery
  - Recreate AP VIF: iw dev ap0 del; iw dev wlp0s20f3 interface add ap0 type __ap; nmcli dev set ap0 managed yes; nmcli con up FieldLAN ifname ap0.
  - Bounce wg: wg-quick down wg0; wg-quick up wg0.

## Security/Privacy

- AP uses WPA2-PSK; PSK length validated (≥ 8).
- Traffic from AP clients traverses WireGuard; NAT applied at wg0.
- DNS for AP clients can be set per FieldLAN to avoid system/global resolver side effects.
- Data protection: CGU notes IAM may process and share customer data with partners; customer has rights under Law 09‑08 to access/rectify/object.

## Compliance notes

- CGU consulted (Power Fibre). No explicit prohibition of VPN found; common prohibitions are resale/redistribution/public offering and unlawful use.
- Demo is private, non-commercial, limited scope; low compliance risk.
- For sustained off‑site use, obtain written confirmation from ISP; comply with workplace security policy.

## Open issues / Future work

- Automate STA channel tracking to reconfigure AP on roam/channel change.
- Optional: move to dual‑radio for higher reliability/throughput.
- Observability: lightweight health check for NM/dnsmasq/wg with auto-restart.

## Decision record

This ADR documents the agreed approach for the demo and near‑term field needs. Changes (e.g., switching to dual‑radio or full “all traffic via VPN”) should be captured in a follow‑up ADR.