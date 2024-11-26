import os
from scapy.all import rdpcap
from loguru import logger
from joblib import Parallel, delayed
from typing import List, Dict, Optional

class PcapProcessor:
    def __init__(
        self, 
        log_level: str = "INFO", 
        log_file: Optional[str] = None
    ):
        """
        Initialize the PCAP processor with logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for log output
        """
        # Configure logging
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr, 
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Optional file logging
        if log_file:
            logger.add(
                log_file, 
                rotation="10 MB",
                level=log_level
            )
        
        self.logger = logger

    def process_packet(
        self, 
        packet, 
        verbose: bool = False
    ) -> Optional[Dict]:
        """
        Process an individual packet.
        
        Args:
            packet: Scapy packet object
            verbose: Boolean flag to control output detail
        
        Returns:
            dict: A summary of the packet or None
        """
        try:
            packet_summary = {
                'summary': str(packet.summary()),
                'layers': [layer.name for layer in packet.layers()],
                'time': packet.time if hasattr(packet, 'time') else None
            }
            
            if verbose:
                self.logger.debug(f"Packet Summary: {packet_summary}")
            
            return packet_summary
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
            return None

    def parallel_pcap_process(
        self, 
        file_path: str, 
        num_workers: Optional[int] = None, 
        packets_per_chunk: int = 1000, 
        verbose: bool = False
    ) -> List[Dict]:
        """
        Parallelize PCAP file processing using joblib.
        
        Args:
            file_path: Path to the .cap file
            num_workers: Number of CPU cores to use
            packets_per_chunk: Number of packets to process in each chunk
            verbose: Boolean flag to control output detail
        
        Returns:
            list: Processed packet summaries
        """
        # Validate input file
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"PCAP file not found: {file_path}")
        
        # If no workers specified, use number of logical cores
        if num_workers is None:
            num_workers = os.cpu_count()
        
        # Load packets
        self.logger.info(f"Loading packets from {file_path}...")
        try:
            packets = rdpcap(file_path)
        except Exception as e:
            self.logger.error(f"Error reading PCAP file: {e}")
            raise
        
        total_packets = len(packets)
        self.logger.info(f"Total packets: {total_packets}")
        
        # Split packets into chunks
        packet_chunks = [
            packets[i:i + packets_per_chunk] 
            for i in range(0, total_packets, packets_per_chunk)
        ]
        
        # Parallel processing using joblib
        try:
            processed_packets = Parallel(n_jobs=num_workers)(
                delayed(self.process_batch)(chunk, verbose) 
                for chunk in packet_chunks
            )
        except Exception as e:
            self.logger.error(f"Parallel processing error: {e}")
            raise
        
        # Flatten results
        processed_packets = [
            pkt for chunk in processed_packets 
            for pkt in chunk if pkt is not None
        ]
        
        self.logger.success(f"Processed {len(processed_packets)} packets successfully")
        return processed_packets

    def process_batch(
        self, 
        chunk: List, 
        verbose: bool = False
    ) -> List[Dict]:
        """
        Process a batch of packets.
        
        Args:
            chunk: List of packets
            verbose: Boolean flag to control output detail
        
        Returns:
            list: Processed packet summaries
        """
        return [
            self.process_packet(pkt, verbose) 
            for pkt in chunk if pkt is not None
        ]

def main():
    import sys
    
    # Example usage
    input_file = '/Users/darth/Downloads/packet_00019_20170614105105.cap'
    
    try:
        # Initialize processor with logging
        processor = PcapProcessor(
            log_level="DEBUG",  # Adjust log level as needed
            log_file="pcap_processing.log"  # Optional log file
        )
        
        # Process the file with verbose output for the first 5 packets
        processed_packets = processor.parallel_pcap_process(
            input_file, 
            num_workers=None,  # Use all available cores
            packets_per_chunk=1000,  # Adjust based on memory constraints
            verbose=True  # Set to False for less output
        )
        
        # Optional: Further analysis or saving results
        # For example, printing first 5 processed packets
        for i, pkt in enumerate(processed_packets[:5], 1):
            processor.logger.info(f"Processed Packet {i}: {pkt}")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
