# Constant for RELU^2 scaling
RELU2_SCALE = 1

# Size of blocks
BLOCK_M = 64        # Rows per tile
BLOCK_N = 64        # Columns per tile

# Number of tiles per chunk used for temporary staging while packing the final bitstream.
_PACK_CHUNK_TILES = 2**16
