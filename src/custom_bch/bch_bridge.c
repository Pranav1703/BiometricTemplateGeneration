#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

// 1. Manually declare the Linux functions with the CORRECT signature
struct bch_control {
    unsigned int m;
    unsigned int n;
    unsigned int t;
    unsigned int ecc_bits;
    unsigned int ecc_bytes;
    // ... hidden internal fields ...
};

// NOTICE: Added 'bool swap_bits' to match bch.c
struct bch_control *bch_init(int m, int t, unsigned int prim_poly, bool swap_bits);
void bch_encode(struct bch_control *bch, const uint8_t *data, unsigned int len, uint8_t *ecc);
int bch_decode(struct bch_control *bch, const uint8_t *data, unsigned int len, const uint8_t *recv_ecc, const uint8_t *calc_ecc, const unsigned int *syn, unsigned int *errloc);
void bch_free(struct bch_control *bch);

// 2. Windows requires explicit EXPORT tags for DLLs
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// 3. Our Custom Wrappers
EXPORT struct bch_control* bch_init_custom(int m, int t) {
    // 0 for default poly, false for swap_bits
    return bch_init(m, t, 0, false); 
}

EXPORT void bch_encode_custom(struct bch_control* bch, uint8_t* data, unsigned int data_len, uint8_t* ecc) {
    bch_encode(bch, data, data_len, ecc);
}

EXPORT int bch_decode_custom(struct bch_control* bch, uint8_t* data, unsigned int data_len, uint8_t* ecc) {
    // Allocate space to hold the error locations
    unsigned int* errloc = (unsigned int*)malloc(bch->t * sizeof(unsigned int));
    
    // Find errors (Pass NULL for calc_ecc and syn)
    int nerr = bch_decode(bch, data, data_len, ecc, NULL, NULL, errloc);
    
    if (nerr > 0) {
        unsigned int data_bits = data_len * 8;
        unsigned int ecc_bits = bch->ecc_bytes * 8;
        
        // Fast C bit-flipping
        for (int i = 0; i < nerr; i++) {
            if (errloc[i] < data_bits) {
                // Error is in the Secret Key
                data[errloc[i] >> 3] ^= (1 << (errloc[i] & 7));
            } else if (errloc[i] < (data_bits + ecc_bits)) {
                // Error is in the Parity Bits
                ecc[(errloc[i] - data_bits) >> 3] ^= (1 << ((errloc[i] - data_bits) & 7));
            }
        }
    }
    
    free(errloc);
    return nerr;
}

EXPORT void bch_free_custom(struct bch_control* bch) {
    bch_free(bch);
}