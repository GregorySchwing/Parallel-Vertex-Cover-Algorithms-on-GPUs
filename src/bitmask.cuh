__constant__ static unsigned char charmask[] = {128, 64, 32, 16, 8, 4, 2, 1};
__constant__ static unsigned int uintmask[] = { 2147483648, 
                                                1073741824,
                                                536870912, 
                                                268435456,
                                                134217728,
                                                67108864,
                                                33554432,
                                                16777216,
                                                8388608,
                                                4194304,
                                                2097152,
                                                1048576,
                                                524288,
                                                262144,
                                                131072,
                                                65536,
                                                32768,
                                                16384,
                                                8192,
                                                4096,
                                                2048,
                                                1024,
                                                512,
                                                256,
                                                128, 64, 32, 16, 8, 4, 2, 1};

inline __device__ int isNthBitSet (unsigned char c, int n) {
    return ((c & charmask[n]) != 0);
}

inline __device__ int isNthBitSet (unsigned int c, int n) {
    return ((c & charmask[n]) != 0);
}


inline __device__ int setNthBit (unsigned char c, int n) {
    return (c | charmask[n]);
}

inline __device__ int setNthBit (unsigned int c, int n) {
    return (c | uintmask[n]);
}