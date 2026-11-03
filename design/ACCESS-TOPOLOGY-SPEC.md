# Access topology specification

Every node receives the same reachable-network inventory. The inventory is the gitignored
`networks.conf`, installed as `/usr/local/mesh/networks.conf` at mode `0600`. Each nonempty
row is a literal tab-separated SSID, security mode, and password. No other credential cache,
dialogue, machine profile, or address roster defines network membership.

`bin/mesh-networks.sh` transduces every row into the host network manager. On macOS it updates
the system Wi-Fi credential and preferred-network set. On NetworkManager systems it creates or
updates the matching connection profile. Both paths turn the radio capability on, preserve the
active connection, retain all previously reachable networks, and realize the complete input
inventory. Provisioning supplies the file before invoking this transducer.

Reachability is a graph, not a preferred address. Wi-Fi, wired LAN, tailnet, Thunderbolt RDMA,
and point-to-point workstation subnet edges coexist. Discovery supplies current addresses;
stable node names select nodes. A workstation on the same WLAN as a stationary node may carry
that node through its independent tailnet and point-to-point edges. Removing or moving a laptop
therefore cannot remove the stationary subgraph.

The implementation is substantiated by the inventory-row count, realized-profile count,
active-interface identity, current-address discovery, and an SSH reachability measure for every
edge used to connect the graph. A password prompt, a fixed DHCP address as canonical identity,
an uninstalled inventory row, or a route whose only carrier is a departing node contradicts
this specification.
