import os
import numpy as np
import pandas as pd
import scapy.all as scapy
from loguru import logger
from scipy import stats
import joblib

class CIPPacketParser:
    def __init__(self, pcap_directory):
        """
        Initialize parser with directory containing PCAP files
        
        Args:
            pcap_directory (str): Path to directory with PCAP files
        """
        self.pcap_directory = pcap_directory
        self.parsed_data = {}
        logger.add("cip_parsing.log", rotation="10 MB")

    def _extract_cip_packets(self, pcap_file):
        """
        Extract CIP packets from a single PCAP file
        
        Args:
            pcap_file (str): Path to PCAP file
        
        Returns:
            list: Extracted CIP packets
        """
        try:
            packets = scapy.rdpcap(pcap_file)
            cip_packets = [
                pkt for pkt in packets 
                if scapy.TCP in pkt and pkt[scapy.TCP].dport == 44818  # Standard CIP port
            ]
            return cip_packets
        except Exception as e:
            logger.error(f"Error extracting packets from {pcap_file}: {e}")
            return []

    def _parse_cip_payload(self, packet):
        """
        Parse CIP packet payload for physical readings
        
        Args:
            packet (scapy.Packet): CIP network packet
        
        Returns:
            dict: Parsed physical readings
        """
        try:
            # Implement specific CIP payload parsing logic
            # This is a placeholder and needs customization based on your specific CIP implementation
            payload = packet[scapy.Raw].load
            # Add your specific parsing logic here
            # Extract sensor names, values, timestamps
            return {
                'sensor_name': 'example_sensor',
                'value': float(payload[:4].hex(), 16),
                'timestamp': packet.time
            }
        except Exception as e:
            logger.warning(f"Payload parsing error: {e}")
            return None

    def parse_network_data(self):
        """
        Parse all PCAP files in the directory
        """
        pcap_files = [
            os.path.join(self.pcap_directory, f) 
            for f in os.listdir(self.pcap_directory) 
            if f.endswith('.pcap')
        ]

        # Use joblib for parallel processing
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._process_pcap_file)(pcap_file) 
            for pcap_file in pcap_files
        )

        # Combine results
        for result in results:
            self.parsed_data.update(result)

    def _process_pcap_file(self, pcap_file):
        """
        Process a single PCAP file
        
        Args:
            pcap_file (str): Path to PCAP file
        
        Returns:
            dict: Parsed data from the file
        """
        logger.info(f"Processing file: {pcap_file}")
        cip_packets = self._extract_cip_packets(pcap_file)
        file_data = {}

        for packet in cip_packets:
            parsed_reading = self._parse_cip_payload(packet)
            if parsed_reading:
                sensor = parsed_reading['sensor_name']
                if sensor not in file_data:
                    file_data[sensor] = []
                file_data[sensor].append(parsed_reading['value'])

        return file_data

    def compute_statistics(self):
        """
        Compute statistics for parsed data
        
        Returns:
            dict: Computed statistics for each sensor
        """
        statistics = {}
        for sensor, values in self.parsed_data.items():
            statistics[sensor] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std_dev': np.std(values)
            }
        return statistics

    def compare_with_original_dataset(self, original_dataset):
        """
        Compare parsed data with original dataset
        
        Args:
            original_dataset (dict): Statistics from original dataset
        """
        parsed_stats = self.compute_statistics()
        comparison_results = {}

        for sensor in set(parsed_stats.keys()) | set(original_dataset.keys()):
            parsed_sensor_stats = parsed_stats.get(sensor, {})
            original_sensor_stats = original_dataset.get(sensor, {})

            comparison_results[sensor] = {
                'parsed_stats': parsed_sensor_stats,
                'original_stats': original_sensor_stats,
                'differences': {
                    metric: abs(parsed_sensor_stats.get(metric, 0) - 
                               original_sensor_stats.get(metric, 0))
                    for metric in ['mean', 'median', 'std_dev']
                }
            }

        return comparison_results

def main():
    PCAP_DIRECTORY = '/path/to/your/pcap/files'
    
    # Example original dataset statistics (you'd load this from your reference)
    original_dataset_stats = {
        'sensor1': {'mean': 50.0, 'median': 49.5, 'std_dev': 5.0},
        # Add more sensors
    }

    parser = CIPPacketParser(PCAP_DIRECTORY)
    parser.parse_network_data()
    
    # Compute and print statistics
    parsed_stats = parser.compute_statistics()
    logger.info("Parsed Statistics: ", parsed_stats)
    
    # Compare with original dataset
    comparison = parser.compare_with_original_dataset(original_dataset_stats)
    logger.info("Comparison Results: ", comparison)

if __name__ == "__main__":
    main()