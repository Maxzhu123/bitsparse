import torch
from nvidia import nvcomp
import time

algos = ["LZ4", "Snappy", "Zstd", "Cascaded", "Deflate", "GDeflate", "ANS", "Bitcomp", "Gzip"]


class Compressor:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def compress_tensor(self, x: torch.Tensor):
        assert x.is_cuda
        assert x.numel() > 0

        x = x.detach().contiguous()
        raw = x.view(torch.uint8).reshape(-1)

        stream = torch.cuda.current_stream(x.device).cuda_stream
        device_id = x.device.index

        codec = nvcomp.Codec(
            algorithm=self.algorithm,
            device_id=device_id,
            cuda_stream=stream,
        )

        src = nvcomp.as_array(raw, cuda_stream=stream)

        # Let PyTorch own the compressed allocation.
        max_size = codec.get_max_comp_buffer_size(src)

        storage = torch.empty(
            max_size,
            dtype=torch.uint8,
            device=x.device,
        )

        dst = nvcomp.as_array(storage, cuda_stream=stream)

        compressed_nv = codec.encode(src, out=dst)

        # Only this prefix contains valid compressed data.
        compressed = storage[:compressed_nv.buffer_size].clone()

        meta = {
            "shape": x.shape,
            "dtype": x.dtype,
            "nbytes": raw.numel(),
            "algorithm": self.algorithm,
        }

        return compressed, meta


    def decompress_tensor(self, compressed, meta):
        assert compressed.is_cuda

        device = compressed.device
        stream = torch.cuda.current_stream(device).cuda_stream

        codec = nvcomp.Codec(algorithm=meta["algorithm"], device_id=device.index, cuda_stream=stream)

        # This should now work, because compressed is backed by
        # a normal PyTorch CUDA allocation.
        compressed_nv = nvcomp.as_array(compressed, cuda_stream=stream)

        raw = torch.empty(meta["nbytes"], dtype=torch.uint8, device=device)

        raw_nv = nvcomp.as_array(raw, cuda_stream=stream,)

        codec.decode(compressed_nv,out=raw_nv,)

        return raw.view(meta["dtype"]).reshape(meta["shape"])


def main():
    x = torch.randn((10000, 21504),
        device="cuda", dtype=torch.bfloat16,
    )
    x = x * (x>0)

    for algo in algos:
        # Warmup
        torch.cuda.empty_cache()
        compressor = Compressor(algorithm=algo)
        for _ in range(2):
            compressed, meta = compressor.compress_tensor(x)
            y = compressor.decompress_tensor(compressed, meta)
        torch.cuda.synchronize()

        st = time.perf_counter()
        for _ in range(2):
            compressed, meta = compressor.compress_tensor(x)
            y = compressor.decompress_tensor(compressed, meta)
            # del compressed, meta
        torch.cuda.synchronize()
        end = time.perf_counter()

        print(f"{algo}:")
        print("original:", x.nbytes//1024**2, "compressed:", compressed.numel()//1024**2)
        print(f'Time: {(end - st)/2:.4f} seconds')

        assert torch.equal(x, y)


if __name__ == "__main__":
    main()
