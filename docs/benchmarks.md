# Benchmarks

## Benchmark Environment

- CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
  - Core Count: 40
- RAM: 256GB
- GPU: NVIDIA TITAN V

## Results

| implementation \ throughput     | secp256k1 point add | secp256k1 scalar mul |
| ------------------------------- | ------------------- | -------------------- |
| OpenSSL 3.0.8 (single thread)   | 996.36 Kop/s        | 3.12 Kop/s           |
| GEC 0.1.0 (cpu) (single thread) | 1311.32 Kop/s       | 3.19 Kop/s           |
| GEC 0.1.0 (cuda)                | 346015.35 Kop/s     | 1389.33 Kop/s        |

## Benchmark Outputs

```plaintext
$ ./build/bench/bench_openssl 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bench_openssl is a Catch v2.13.8 host application.
Run with -? for options

-------------------------------------------------------------------------------
openssl
-------------------------------------------------------------------------------
/home/zjn/gec/bench/bench_openssl.cpp:5
...............................................................................

benchmark name                       samples       iterations    estimated
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
secp256k1 point add                            100            24      2.448 ms 
                                        1.00365 us    1.00012 us    1.00739 us 
                                        18.5245 ns    16.2216 ns    21.5301 ns 
                                                                               
secp256k1 scalar mul                           100             1    32.1096 ms 
                                        320.942 us     320.27 us    321.602 us 
                                        3.40034 us    2.86878 us    4.25078 us 

===============================================================================
All tests passed (19 assertions in 1 test case)

$ ./build/bench/bench_gec_cpu 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bench_gec_cpu is a Catch v2.13.8 host application.
Run with -? for options

-------------------------------------------------------------------------------
gec_cpu
-------------------------------------------------------------------------------
/home/zjn/gec/bench/bench_gec_cpu.cpp:5
...............................................................................

benchmark name                       samples       iterations    estimated
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
secp256k1 point add                            100            32     2.3584 ms 
                                        762.597 ns    760.525 ns    765.554 ns 
                                        12.4768 ns    9.25585 ns     16.329 ns 
                                                                               
secp256k1 scalar mul                           100             1    28.4693 ms 
                                        313.415 us    311.581 us    315.347 us 
                                        9.61369 us    8.34086 us    11.5316 us 
                                                                               

===============================================================================
test cases: 1 | 1 passed
assertions: - none -

$ ./build/bench/bench_gec_cuda 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bench_gec_cuda is a Catch v2.13.8 host application.
Run with -? for options

-------------------------------------------------------------------------------
gec_cuda
-------------------------------------------------------------------------------
/home/zjn/gec/bench/bench_gec_cuda.cu:67
...............................................................................

benchmark name                       samples       iterations    estimated
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
secp256k1 point add x655360                    100             1    195.504 ms 
                                        1.89402 ms    1.88999 ms    1.89814 ms 
                                        20.7388 us    18.6913 us    23.2618 us 
                                                                               
secp256k1 scalar mul x491520                   100             1      35.362 s 
                                        353.782 ms    353.707 ms    353.854 ms 
                                        372.965 us    327.222 us     433.94 us 
                                                                               

===============================================================================
All tests passed (428 assertions in 1 test case)
```