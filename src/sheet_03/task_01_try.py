import pyshark

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

def get_vendor_by_mac(mac_address):
    # Extract the first 3 bytes (OUI) from the MAC address
    mac_oui = ':'.join(mac_address.split(':')[:3]).upper()
    return OUI_MAP.get(mac_oui, 'Unknown Vendor')

# Load the .cap file
file_path = 'first100_00173.cap'
capture = pyshark.FileCapture(file_path)

for packet in capture:
    if 'CIP' in packet:
        #print(packet)
        #print(dir(packet))
        #print(packet.cip)########################
        #print(dir(packet.cip))
        #print(f"Raw CIP Fields: {packet.cip._all_fields}")
        #print(f"\n\n\n\nRaw Fields: {dir(packet.cipcm)}\n\n\n\n\n")

        print(f"Packet Number: {packet.number}")
        print(f"Timestamp: {packet.sniff_time}")
        print(f"Source IP: {packet.ip.src if 'IP' in packet else 'N/A'}")
        print(f"Destination IP: {packet.ip.dst if 'IP' in packet else 'N/A'}")
        print(f"Protocol: {packet.highest_layer}")
        
        try:
            src_mac = packet.eth.src if 'ETH' in packet else 'N/A'
            vendor_name = get_vendor_by_mac(src_mac)
            print(f"Device Vendor (from MAC OUI): {vendor_name} ({src_mac})")
            # Parse CIP Service
            #service_code = int(packet.cip.service, 16) if hasattr(packet.cip, 'service') else None
            service_code = int(packet.cip.sc, 16) if hasattr(packet.cip, 'service') else None
            if service_code is not None:
                service_description = CIP_SERVICE_MAP.get(service_code, f"Unknown Service (0x{service_code:02X})")
                print(f"CIP Service: {service_description}")
            else:
                print("CIP Service: Not found")

            request_path = None
            try: request_path = packet.cip.symbol
            except: pass
            try: request_path = packet.cipcm.cip_symbol
            except: pass
            if request_path != None:
                print(f"CIP Request Path: {request_path}")
            else:
                print("CIP Request Path: Not found")

            data = None
            #print("hhhhhhhhhhhhhhh:",(packet.cipcls.cip_data))
            #print("hhhhhhhhhhhhhhh:",dir(packet.cipcls))
            try: data = packet.cipcm.cip_data
            except: pass
            try: data = packet.cipcls.cip_data
            except: pass
            if data != None: 
                data
                print(f"Data: {data}")
            else:
                print("Data: Not found")
            
        except AttributeError as e:
            print("Error accessing CIP attributes:", e)

        print("-" * 50)
capture.close()


# Optionally, save the parsed data to a file or data structure for analysis.
# store physical readings, name of the sensor / actuator, corresponding (numerical) values. 
#htop
#pip freeze -> generate requirements

#udp port 2222
#tcp port 44818 -------tcp


