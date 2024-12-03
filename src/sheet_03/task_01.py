from scapy.all import *
from collections import defaultdict
import pandas as pd
from datetime import datetime

class SWaTParser:
    def __init__(self):
        self.data_points = defaultdict(list)
        self.timestamps = []
        
        # Define sensor and actuator columns based on SWaT structure
        self.columns = [
            'Time',
            'SWAT_SUTD:RSLinx Enterprise:P1.HMI_FIT101.Pv',
            'SWAT_SUTD:RSLinx Enterprise:P1.HMI_LIT101.Pv',
            # Add all other columns as per example
        ]
        
    def process_packet(self, packet):
        if not (UDP in packet and packet[UDP].sport == 2222 and packet[UDP].dport == 2222):
            return
            
        timestamp = packet.time
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y %I:%M:%S.%f %p')
        
        if Raw in packet:
            raw_data = packet[Raw].load
            
            # Skip EtherNet/IP header
            # Type ID: Sequenced Address Item (0x8002)
            # Type ID: Connected Data Item (0x00b1)
            data_offset = 4  # Adjust based on actual header size
            
            # Extract CIP I/O data
            cip_data = raw_data[data_offset:]
            
            # Parse the data into sensor/actuator values
            readings = self.parse_cip_values(cip_data)
            
            if readings:
                self.timestamps.append(formatted_time)
                for key, value in readings.items():
                    self.data_points[key].append(value)
    
    def parse_cip_values(self, cip_data):
        readings = {}
        
        # Example parsing logic - adjust based on actual data format
        try:
            # Parse hex data into corresponding sensor/actuator values
            # Example: bcaf01000000ffffffff000000000000000000000000000000
            
            # Each sensor/actuator value might occupy specific positions
            # in the data string with specific lengths
            
            readings['SWAT_SUTD:RSLinx Enterprise:P1.HMI_FIT101.Pv'] = float(self.extract_value(cip_data, 0, 4))
            readings['SWAT_SUTD:RSLinx Enterprise:P1.HMI_LIT101.Pv'] = float(self.extract_value(cip_data, 4, 4))
            # Add parsing for other sensors/actuators
            
        except Exception as e:
            print(f"Error parsing CIP data: {e}")
            return None
            
        return readings
    
    def extract_value(self, data, start, length):
        # Extract and convert binary data to appropriate format
        # This is a placeholder - implement actual conversion logic
        value_bytes = data[start:start+length]
        return int.from_bytes(value_bytes, byteorder='little')
    
    def create_dataframe(self):
        df = pd.DataFrame(self.data_points, index=self.timestamps)
        df.index.name = 'Time'
        return df
    
    def save_to_csv(self, filename):
        df = self.create_dataframe()
        df.to_csv(filename)

def main():
    parser = SWaTParser()
    
    # Process pcap files
    pcap_files = ['your_pcap_file.pcap']  # Add your pcap files
    
    for pcap_file in pcap_files:
        packets = PcapReader(pcap_file)
        for packet in packets:
            parser.process_packet(packet)
    
    # Save results
    parser.save_to_csv('parsed_swat_data.csv')
    
    # Calculate statistics
    df = pd.read_csv('parsed_swat_data.csv')
    statistics = df.describe()
    statistics.to_csv('statistics.csv')

if __name__ == "__main__":
    main()