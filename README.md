# d-HNSW: Efficient Vector Search on Disaggregated Memory

The first vector search engine
for RDMA-based disaggregated memory systems. 


## Project Structure

```
d-HNSW/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── dhnsw/              # Core library 
│   ├── generated/          # Protobuf/gRPC generated code a
│   └── bench/              # benchmarks, tests
├── deps/                   # Dependencies
├── xcomm/                  # RDMA/RPC communication library
├── util/                  
├── scripts/                
├── tools/                  # Data conversion utilities 
├── tests/                  
└── docs/                   # Documentation
```

## Prerequisites

- **OS**: Ubuntu 18.04 or 20.04
- **Hardware**: Mellanox InfiniBand NIC (ConnectX-3 or later)
- **Compiler**: GCC with C++17 support
- **RDMA**: Mellanox OFED driver
- **Libraries**: CMake >= 3.5, Boost (coroutine, context), OpenMP, BLAS/LAPACK, Protobuf, gRPC, gflags, libibverbs

## Building

### 1. System Setup

Install Mellanox OFED, build tools, and configure huge pages (run on **every** node):

```bash
git clone --recursive https://github.com/fffeifang/d-HNSW.git
cd d-HNSW

# Skip disk mount (if already set up):
sudo ./scripts/setup.sh
```

A system reboot is required after setup.

### 2. Install Dependencies

This builds all dependencies from `deps/` submodules (GKlib, METIS, gflags, Abseil, gRPC v1.67). Building gRPC from source takes 10-30 minutes but is the recommended approach:

```bash
./scripts/install_dependencies.sh
```

### 3. Download Datasets

```bash
./scripts/datasets.sh
```

This downloads standard ANN benchmarks: SIFT1M, GIST1M, SIFT10M, DEEP10M, TEXT10M.

### 4. Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Build targets:

| Target | Description |
|--------|-------------|
| `run_server` | Memory server |
| `run_client` | Compute node |

### RDMA Network Configuration

Assign an IP address to the InfiniBand interface on each node. The interface name depends on your hardware (check with `ibstat` or `ip link`).

```bash
# Memory server (e.g., 192.168.1.2)
sudo ifconfig <ib_interface> 192.168.1.2 netmask 255.255.0.0

# Compute node 1 (e.g., 192.168.1.10)
sudo ifconfig <ib_interface> 192.168.1.10 netmask 255.255.0.0

# Compute node 2 (e.g., 192.168.1.11)
sudo ifconfig <ib_interface> 192.168.1.11 netmask 255.255.0.0

# Compute node 3 (e.g., 192.168.1.11)
sudo ifconfig <ib_interface> 192.168.1.12 netmask 255.255.0.0
```

Common interface names:  `ibp8s0` (ConnectX-3), `ens2f0` (ConnectX-5/6).

### Verify RDMA Connectivity

Run `ib_write_bw` between the memory server and each compute node:

```bash
# On memory server
ib_write_bw -d <mlx_device> -i 1 -D 10 --report_gbits

# On compute node
ib_write_bw 192.168.1.2 -d <mlx_device> -i 1 -D 10 --report_gbits
```

Check the device name with `ibstat` (e.g., `mlx4_0`, `mlx5_0`, `mlx5_2`).


### Find Your NIC Index

d-HNSW needs the NIC index (`--use_nic_idx`) to create RDMA queue pairs.

Use the NIC index corresponding to your machine configuration. For example:

-	CloudLab r650 w./ Mlnx CX6 100 Gb NIC (~92.57 Gbits): --use_nic_idx=3

-	CloudLab r320 w./ Mlnx MX354A FDR CX3 adapter (~55.52 Gbits): --use_nic_idx=0

## Quick Start

### Example: 2-Node Deployment

Suppose the memory server has gRPC IP `10.0.0.1` and RDMA IP `192.168.1.2`.

**Node 1 -- Memory server:**

```bash
cd build
./run_server \
    --server_ip=10.0.0.1 \
    --port=50051 \
    --rdma_port=8888 \
    --use_nic_idx=3 \
    --dataset_path=../datasets/sift/sift_base.fvecs \
    --dim=128 \
    --num_sub_hnsw=160 \
    --num_meta=5000
```

**Node 2 -- Compute node (insert + search):**

```bash
cd build
./run_client \
    --server_address=10.0.0.1:50051 \
    --rdma_server_address=192.168.1.2:8888 \
    --use_nic_idx=3 \
    --dataset=sift1M
```

## Fine-tuning Dataset Parameters

Dataset-specific parameters (dimensions, partitioning, search settings) are defined in `src/dhnsw/data_config.hh`. To add a new dataset or adjust existing ones, edit the `config_map`.

## Citation

```bibtex
@inproceedings{liu2025dhnsw,
  author = {Yi Liu and Fei Fang and Chen Qian},
  title = {Efficient Vector Search on Disaggregated Memory with d-HNSW},
  booktitle = {Proceedings of the 17th ACM Workshop on Hot Topics in Storage and File Systems (HotStorage)},
  year = {2025},
  pages = {1--8},
  doi = {10.1145/3736548.3737822}
}

```bibtex
@article{fang2026d,
  title={d-HNSW: A High-performance Vector Search Engine on Disaggregated Memory},
  author={Fang, Fei and Liu, Yi and Qian, Chen},
  journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems},
  volume={10},
  number={2},
  pages={1--28},
  year={2026},
  publisher={ACM New York, NY, USA}
}
