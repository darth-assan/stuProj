import scapy.all as scapy
import pandas as pd
import struct
import os
import glob
import numpy as np

def parse_cip_data(raw_payload):
    if len(raw_payload) < 12:
        return None  # Not enough data to unpack

    offset = 24  # Skip EtherNet/IP header
    cmd, length, session, status = struct.unpack('<HHII', raw_payload[0:12])
    
    # Parse CIP data
    if cmd == 0x0070 or cmd == 0x006f:  # SendRRData or SendUnitData
        item_count = struct.unpack('<H', raw_payload[30:32])[0]
        offset = 32
        
        for _ in range(item_count):
            type_id, item_len = struct.unpack('<HH', raw_payload[offset:offset+4])
            offset += 4
            
            if type_id == 0x00b1:  # Connected Data Item
                seq_cnt = struct.unpack('<H', raw_payload[offset:offset+2])[0]
                offset += 2
                
                service = struct.unpack('<B', raw_payload[offset:offset+1])[0] & 0x7F
                offset += 1
                
                if service == 0x4C:  # Read Tag Service
                    data_type = struct.unpack('<H', raw_payload[offset:offset+2])[0]
                    offset += 2
                    
                    if data_type == 0xCA:  # REAL
                        value = struct.unpack('<f', raw_payload[offset:offset+4])[0]
                        return value
    
    return None

def parse_pcap(file_path):
    readings = []
    packet_count = 0
    valid_readings = 0
    
    for packet in scapy.PcapReader(file_path):
        packet_count += 1
        if scapy.TCP in packet and scapy.Raw in packet:
            src_port = packet[scapy.TCP].sport
            dst_port = packet[scapy.TCP].dport
            
            if src_port == 44818 or dst_port == 44818:  # EtherNet/IP port
                raw_payload = bytes(packet[scapy.Raw].load)
                value = parse_cip_data(raw_payload)
                
                if value is not None:
                    valid_readings += 1
                    readings.append({
                        'timestamp': packet.time,
                        'value': value
                    })
    
    print(f"Processed {packet_count} packets, found {valid_readings} valid readings")
    return pd.DataFrame(readings)

def process_pcap_files(directory):
    all_readings = []
    
    for file in glob.glob(os.path.join(directory, '*.pcap')):
        print(f"Processing {file}...")
        df = parse_pcap(file)
        if not df.empty:  # Only append non-empty DataFrames
            all_readings.append(df)
        else:
            print(f"No valid readings found in {file}")
    
    if not all_readings:  # Check if we have any readings at all
        print("No valid readings found in any files")
        return pd.DataFrame(columns=['timestamp', 'value'])  # Return empty DataFrame with correct columns
    
    return pd.concat(all_readings).sort_values('timestamp').reset_index(drop=True)

def compute_statistics(df):
    return {
        'mean': df['value'].mean(),
        'median': df['value'].median(),
        'std': df['value'].std()
    }

# Main execution
pcap_directory = '/Users/darth/Data/stuProj/SWaT/test'
parsed_readings = process_pcap_files(pcap_directory)

# Save parsed readings to CSV
parsed_readings.to_csv('parsed_readings.csv', index=False)

# Compute statistics
parsed_stats = compute_statistics(parsed_readings)

print("Statistics for parsed readings:")
print(parsed_stats)

# Compare with provided dataset (assuming it's in a CSV file)
provided_readings = pd.read_csv('provided_readings.csv')
provided_stats = compute_statistics(provided_readings)

print("\nStatistics for provided readings:")
print(provided_stats)

print("\nComparison:")
for stat in ['mean', 'median', 'std']:
    diff = abs(parsed_stats[stat] - provided_stats[stat])
    print(f"{stat.capitalize()} difference: {diff}")