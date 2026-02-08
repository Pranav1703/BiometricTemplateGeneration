from reedsolo import RSCodec, ReedSolomonError

class ECCWrapper:
    """
    A wrapper around Reed-Solomon error correction for consistent usage.
    """
    def __init__(self, message_size: int = 32, error_capacity_percent: float = 0.2):
        """
        Initialize ECC wrapper.

        Args:
            message_size (int): The size of message in bytes (default 32 bytes for 256 bits).
            error_capacity_percent (float): The percentage of errors to correct (default 20%).
        """
        self.message_size = message_size
        
        # Calculate required parity bytes. 
        # Reed-Solomon can correct (n_parity / 2) errors.
        # So to correct X bytes, we need 2*X parity bytes.
        # We want to correct error_capacity_percent * block_size? 
        # Usually capacity is relative to total block size or message size.
        # Let's interpret "20% error-correction capacity" as correcting 20% of total block.
        # But commonly we define it based on message size for simple configuration.
        # If we target correcting `k` errors, we need `2*k` parity bytes.
        # Total size = message_size + 2*k.
        # If k is approx 20% of total size... 
        # Let's calculate parity length such that we can correct roughly the requested percentage of the total packet.
        # Let T be total length, M be message length, P be parity length. T = M + P.
        # Correctable errors E = P / 2.
        # We want E / T >= error_capacity_percent.
        # (P/2) / (M + P) >= target
        # P / (2M + 2P) >= target
        # P >= target * (2M + 2P)
        # P >= 2*target*M + 2*target*P
        # P * (1 - 2*target) >= 2*target*M
        # P >= (2*target*M) / (1 - 2*target)
        
        # However, a simpler heuristic often used is just adding parity bytes relative to message size.
        # If we just simply allow correcting `round(message_size * error_capacity_percent)` errors:
        # For 32 bytes, 20% is 6.4 bytes -> 6 bytes. 
        # So we need 12 parity bytes.
        # This seems reasonable.
        
        errors_to_correct = int(message_size * error_capacity_percent)
        # Ensure at least 1 byte correction if any percent is given
        if errors_to_correct == 0 and error_capacity_percent > 0:
            errors_to_correct = 1
            
        n_ec_bytes = errors_to_correct * 2
        
        self.rsc = RSCodec(n_ec_bytes)

    def encode(self, data: bytes) -> bytes:
        """
        Encodes data with Reed-Solomon error correction.

        Args:
            data (bytes): The input data (must match message_size).

        Returns:
            bytes: The encoded data (message + parity).
        
        Raises:
            ValueError: If data length does not match message_size.
        """
        if len(data) != self.message_size:
            raise ValueError(f"Input data length {len(data)} does not match expected {self.message_size} bytes.")
        
        return bytes(self.rsc.encode(data))

    def decode(self, data: bytes) -> bytes:
        """
        Decodes data and corrects errors if possible.

        Args:
            data (bytes): The encoded data.

        Returns:
            bytes: The original message.
        
        Raises:
            ReedSolomonError: If too many errors are present to correct.
        """
        # decode returns (decoded_message, decoded_message_with_ecc_padding, errata_pos)
        # We just want the message.
        try:
            decoded_msg, _, _ = self.rsc.decode(data)
            return bytes(decoded_msg)
        except ReedSolomonError:
             # Re-raise to let caller handle validity check failure
            raise