from scapy.all import rdpcap
from datetime import datetime

# Path to the uploaded .cap file
cap_file_path = 'first10_00173.cap'

# Read packets from the .cap file
packets = rdpcap(cap_file_path)

# Function to extract and display packet details
# Function to extract and display packet details
def parse_packets(packets):
    print(f"{'No.':<5} {'Timestamp':<20} {'Src MAC':<20} {'Src IP':<15} {'Dst MAC':<20} {'Dst IP':<15} {'Protocol':<10} {'Service':<10}")
    print("="*125)

    for idx, packet in enumerate(packets, start=1):
        # Timestamp
        timestamp = datetime.fromtimestamp(float(packet.time)).strftime('%Y-%m-%d %H:%M:%S')

        
        # MAC Addresses
        src_mac = packet.src if hasattr(packet, 'src') else "N/A"
        dst_mac = packet.dst if hasattr(packet, 'dst') else "N/A"
        
        # IP Addresses
        src_ip = packet['IP'].src if packet.haslayer("IP") else "N/A"
        dst_ip = packet['IP'].dst if packet.haslayer("IP") else "N/A"
        
        # Protocol
        if packet.haslayer("IP"):
            protocol = packet.sprintf("%IP.proto%")
        else:
            protocol = "N/A"
        
        # Service
        if packet.haslayer("TCP"):
            service = f"TCP:{packet['TCP'].sport}->{packet['TCP'].dport}"
        elif packet.haslayer("UDP"):
            service = f"UDP:{packet['UDP'].sport}->{packet['UDP'].dport}"
        else:
            service = "Other"

        print(f"{idx:<5} {timestamp:<20} {src_mac:<20} {src_ip:<15} {dst_mac:<20} {dst_ip:<15} {protocol:<10} {service:<10}")

# Parse and display the packets
parse_packets(packets)
