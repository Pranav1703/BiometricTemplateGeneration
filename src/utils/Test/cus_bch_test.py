import ctypes
import os

class CustomBCH:
    def __init__(self, m, t):
        self.m = m
        self.t = t
        # Calculate parity bytes
        bits = m * t
        self.ecc_bytes = (bits // 8) + 1 if bits % 8 != 0 else bits // 8
        
        # 1. LOAD THE DLL
        # This dynamically finds the DLL as long as it's in the same folder as this script
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libcustombch.dll')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find DLL at: {lib_path}")
            
        print(f"Loading C-Library from: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)
        
        # 2. DEFINE C-TYPES (Memory Safety)
        self.lib.bch_init_custom.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.bch_init_custom.restype = ctypes.c_void_p
        
        self.lib.bch_encode_custom.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.bch_encode_custom.restype = None
        
        self.lib.bch_decode_custom.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.bch_decode_custom.restype = ctypes.c_int
        
        self.lib.bch_free_custom.argtypes = [ctypes.c_void_p]
        self.lib.bch_free_custom.restype = None
        
        # 3. INITIALIZE C-CONTEXT
        self.ctx = self.lib.bch_init_custom(m, t)
        if not self.ctx:
            raise RuntimeError("C library failed to initialize the BCH context.")

    def encode(self, data: bytes):
        data_len = len(data)
        # Convert Python bytes to C arrays
        c_data = (ctypes.c_uint8 * data_len)(*data)
        c_ecc = (ctypes.c_uint8 * self.ecc_bytes)()
        
        # Execute C function
        self.lib.bch_encode_custom(self.ctx, c_data, data_len, c_ecc)
        return bytes(c_ecc)

    def decode(self, data: bytearray, ecc: bytearray):
        data_len = len(data)
        # Convert Python bytearrays to C arrays
        c_data = (ctypes.c_uint8 * data_len)(*data)
        c_ecc = (ctypes.c_uint8 * len(ecc))(*ecc)
        
        # Execute C function (modifies c_data in-place if successful)
        errors_fixed = self.lib.bch_decode_custom(self.ctx, c_data, data_len, c_ecc)
        
        return errors_fixed, bytearray(c_data)

    def __del__(self):
        # Free the C memory when Python deletes this object
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.bch_free_custom(self.ctx)

# --- RUN THE UNIT TEST ---
if __name__ == "__main__":
    print("--- Starting Custom BCH Test ---")
    
    # We want a Galois Field of 12 (max 4095 bits) and t=150 errors!
    bch = CustomBCH(m=12, t=150)
    
    # 1. Create a fake 32-byte (256-bit) Secret Key
    original_secret = b"A" * 32 
    print(f"\n1. Original Secret : {original_secret}")
    
    # 2. Encode to get parity bits
    ecc = bch.encode(original_secret)
    print(f"2. Parity Generated: {len(ecc)} bytes")
    
    # 3. Simulate fingerprint errors (Corrupt the secret)
    # We will flip the first 5 bytes completely (which is ~40 bit errors)
    corrupted_secret = bytearray(original_secret)
    for i in range(5):
        corrupted_secret[i] ^= 0xFF # Flip bits
        
    print(f"3. Corrupted Secret: {bytes(corrupted_secret)}")
    
    # 4. Decode using our C Library
    errors_fixed, recovered_secret = bch.decode(corrupted_secret, bytearray(ecc))
    
    if errors_fixed >= 0:
        print(f"\nSUCCESS! The C library fixed {errors_fixed} errors.")
        print(f"Recovered Secret : {bytes(recovered_secret)}")
        if original_secret == bytes(recovered_secret):
            print("MATCH: The recovered key perfectly matches the original!")
    else:
        print("\nFAILED: Too many errors, could not decode.")