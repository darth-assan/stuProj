import pyshark
import struct
import pandas as pd
from datetime import datetime

print_packet_details = False
print_decoding_details = False
file_path = './first10k_00173.cap'

#https://scholar.google.com.sg/citations?view_op=view_citation&hl=en&user=HvkAJmMAAAAJ&citation_for_view=HvkAJmMAAAAJ:d1gkVwhDpl0C
# Secure Water Treatment system (SWaT)

# AITxxx stands for Analyzer Indicator/Transmitter; 
# DPITxxx stands for Differential Pressure Indicator/Transmitter; 
# FITxxx stands for Flow Indicator Transmitter; 
# LITxxx stands for Level Indicator/Transmitter.
# CIPCM = common industrial protocol connection manager

# OUI Vendor Mapping (Example: Rockwell Automation)
# https://mac.lc/company/rockwell-automation
# Organizationally Unique Identifier (OUI), which is the first 3 bytes (24 bits) of the MAC address

OUI_MAP = {
    '00:00:BC': 'Rockwell Automation',  # OUI for Rockwell Automation
    '00:1D:9C': 'Rockwell Automation',  # OUI for Rockwell Automation
    '00:00:BC': 'Rockwell Automation',  # OUI for Rockwell Automation
    '00:1D:9C': 'Rockwell Automation',  # OUI for Rockwell Automation
    '08:61:95': 'Rockwell Automation',  # OUI for Rockwell Automation
    '18:4C:08': 'Rockwell Automation',  # OUI for Rockwell Automation  EUI-48 (Partial)
    '34:C0:F9': 'Rockwell Automation',  # OUI for Rockwell Automation
    '5C:88:16': 'Rockwell Automation',  # OUI for Rockwell Automation
    'E4:90:69': 'Rockwell Automation',  # OUI for Rockwell Automation
    'F4:54:33': 'Rockwell Automation',  # OUI for Rockwell Automation
}

# CIP Service Codes Mapping
CIP_SERVICE_MAP = {
    # https://docs.pycomm3.dev/en/latest/cip_reference.html
    # Common CIP Services
    b"\x01": "get_attributes_all",
    b"\x02": "set_attributes_all",
    b"\x03": "get_attribute_list",
    b"\x04": "set_attribute_list",
    b"\x05": "reset",
    b"\x06": "start",
    b"\x07": "stop",
    b"\x08": "create",
    b"\x09": "delete",
    b"\x0A": "multiple_service_request",
    b"\x0D": "apply_attributes",
    b"\x0E": "get_attribute_single",
    b"\x10": "set_attribute_single",
    b"\x11": "find_next_object_instance",
    b"\x14": "error_response",
    b"\x15": "restore",
    b"\x16": "save",
    b"\x17": "nop",
    b"\x18": "get_member",
    b"\x19": "set_member",
    b"\x1A": "insert_member",
    b"\x1B": "remove_member",
    b"\x1C": "group_sync",

    # Rockwell Custom Services
    0x4C: "read_tag",
    b"\x52": "read_tag_fragmented",
    0x4D: "write_tag",
    b"\x53": "write_tag_fragmented",
    b"\x4E": "read_modify_write",
    b"\x55": "get_instance_attribute_list",
    # in wireshark, the the CIP service field uses 4 bytes of 4 i.e. 00cd -> 4d (overflow),  also complient to above list
    # in pychark, the the CIP service field uses 2 bytes of 4 i.e. 00cd -> cd,
}

DATA_SIZE_DECODING_MAP = {
    # device name : [length of raw_data, position of value]
    "HMI_AIT402" : [80,4], # or 36
    "HMI_AIT504" : [80,4], # or 36
    "HMI_AIT503" : [80,4], # or 36
    "HMI_AIT502" : [80,4], # or 36
    "HMI_AIT501" : [80,4],  # or 36
    "HMI_LIT101" : [80,4], # or 36
    "HMI_LIT301" : [80,4], # or 36
    "HMI_LIT401" : [80,4] # or 36
}

# Raw data in hex in IEEE 754
def decoding_data(raw_data, request_path):
    if print_decoding_details: print("Device:", request_path)
    if print_decoding_details: print("Raw_data:", raw_data)
    if print_decoding_details: print("Raw_data length:", len(raw_data))
    decoded = "Not decoded"
    
    if request_path != None:
        ####### Manual Analysis #########    
        """
        if request_path == "HMI_LIT401" and len(raw_data) != 4:
            print("Raw data length:", len(raw_data))
            bytes_data = bytes.fromhex(raw_data)
            for i in range(0,len(bytes_data)+1-4):
                new_bytes_data = bytes_data[i:i+4]
                decoded = struct.unpack('<f', new_bytes_data)[0]
                print(i, "=", decoded)
            input("wait here")
        """
        ################
        current = DATA_SIZE_DECODING_MAP.get(request_path, 'Unknown decoding')
        if len(raw_data) == 4:
            decoded = "4 bit output"
        if len(raw_data) == current[0]:
            bytes_data = bytes.fromhex(raw_data)
            new_bytes_data = bytes_data[current[1]:current[1]+4]
            decoded = struct.unpack('<f', new_bytes_data)[0]
    else:
        decoded = "No device name"
    if print_decoding_details: print("Decode result:", decoded)
    return decoded


def add_entry_to_table(request_path, data, base_time):
    global data_table
    if base_time in data_table["Timestamp"].values: #If row exists
        row_index = data_table[data_table["Timestamp"] == base_time].index[0]  # Update the existing row
        if request_path in data_table.columns:   # If col exists
            data_table.at[row_index, request_path] = data   # Update the existing col
        else:   # If col doesnt exists
            data_table[request_path] = None  # create new column
            data_table.at[row_index, request_path] = data   # Update the existing col
    else: # If row doesnt exists
        new_row = {"Timestamp": base_time}  # Create new row
        if request_path not in data_table.columns: # If col doesnt exists
            data_table[request_path] = None  # create new column
        new_row[request_path] = data # update col
        data_table = pd.concat([data_table, pd.DataFrame([new_row])], ignore_index=True)  # enter row


def get_vendor_by_mac(mac_address):
    # Extract the first 3 bytes (OUI) from the MAC address
    mac_oui = ':'.join(mac_address.split(':')[:3]).upper()
    return OUI_MAP.get(mac_oui, 'Unknown Vendor')




# Main
data_table = pd.DataFrame(columns=["Timestamp"])
capture = pyshark.FileCapture(file_path)
for packet in capture:
    if 'CIP' in packet:
        src_ip = "Not found"
        dest_ip = "Not found"
        service_description = "Not found"
        request_path = "Not found"
        vendor_name = "Not found"
        data = "Not found"

        try:
            # Parse source and destination ip
            if 'IP' in packet:
                src_ip = packet.ip.src
                dest_ip = packet.ip.dst
            # Map Vendor name by MAC
            src_mac = packet.eth.src if 'ETH' in packet else 'N/A'
            vendor_name = get_vendor_by_mac(src_mac)
            # Parse CIP Service
            service_code = int(packet.cip.sc, 16) if hasattr(packet.cip, 'service') else None
            if service_code is not None:
                service_description = CIP_SERVICE_MAP.get(service_code, f"Unknown Service (0x{service_code:02X})")
            # Parse requst path
            try: request_path = packet.cip.symbol
            except: pass
            try: request_path = packet.cipcm.cip_symbol
            except: pass
            # Parse data
            try: data = packet.cipcm.cip_data
            except: pass
            try: data = packet.cipcls.cip_data
            except: pass
            # Decode data
            if data != "Not found":
                converted = data.replace(":", "")
                data = decoding_data(converted, request_path)
                if request_path:
                    timestring = (packet.sniff_time).strftime("%Y%m%d%H%M%S") + f"{(packet.sniff_time).microsecond // 1000:03d}"
                    add_entry_to_table(request_path, data, timestring)
            
            print(f"Packet Number: {packet.number}")

            if print_packet_details: print(f"Timestamp: {packet.sniff_time}")
            if print_packet_details: print(f"Source IP: {src_ip}")
            if print_packet_details: print(f"Destination IP: {dest_ip}")
            if print_packet_details: print(f"Protocol: {packet.highest_layer}")
            if print_packet_details: print(f"Device Vendor (from MAC OUI): {vendor_name} ({src_mac})")
            if print_packet_details: print(f"CIP service: {service_description}")
            if print_packet_details: print(f"CIP request path: {request_path}")
            if print_packet_details: print(f"Data: {data}")
        
        except AttributeError as e:
            print("Error accessing CIP attributes:", e)
        print("-" * 50)

capture.close()
print(data_table)
data_table.to_csv('device_data_over_time.csv', index=False)
sorted_columns = sorted([col for col in data_table.columns if col != "Timestamp"]) + ["Timestamp"]
data_table = data_table[sorted_columns]
print(data_table)

# Convert all columns to numeric (floats) where possible, replacing non-convertible values with NaN
numeric_df = data_table.apply(pd.to_numeric, errors='coerce')
print("\nMean:\n", numeric_df.mean())
print("\nMedian:\n",  numeric_df.median())
print("\nStd_dev:\n",  numeric_df.std())


## NOTES to self:
#htop
#pip freeze -> generate requirements

# udp uses port 2222
# tcp uses port 44818
#print(packet)
#print(dir(packet))
#print(packet.cip)
#print(dir(packet.cip))
#print(f"Raw CIP Fields: {packet.cip._all_fields}")
#print(f"\n\n\n\nRaw Fields: {dir(packet.cipcm)}\n\n\n\n\n")



