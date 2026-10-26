| fabric | model | CP convention / batch | best config | step (s) | margin | mem/GPU |
|---|---|---|---|---|---|---|
| 10x8 | llama3-8b | ring all_to_all | tp=1 pp=20 cp=4 dp=1 zero=0 | 0.017 | 0.87% | 7.5 |
| 10x8 | llama3-8b | default all_gather | tp=1 pp=16 cp=1 dp=5 zero=2 | 0.018 | 2.05% | 6.5 |
| 10x8 | llama3-8b | ring all_to_all, batch=8 | tp=5 pp=1 cp=16 dp=1 zero=0 | 0.083 | 0.20% | 26.1 |
| 10x8 | llama3-8b | default all_gather, batch=8 | tp=1 pp=16 cp=5 dp=1 zero=0 | 0.095 | 0.87% | 9.0 |
| 10x8 | llama3-70b | ring all_to_all | tp=1 pp=40 cp=2 dp=1 zero=0 | 0.103 | 4.85% | 31.4 |
| 10x8 | llama3-70b | default all_gather | tp=1 pp=20 cp=1 dp=4 zero=2 | 0.111 | 0.11% | 29.5 |
| 10x8 | llama3-70b | ring all_to_all, batch=8 | tp=1 pp=16 cp=5 dp=1 zero=0 | 0.652 | 0.27% | 73.2 |
| 10x8 | llama3-70b | default all_gather, batch=8 | tp=1 pp=20 cp=4 dp=1 zero=0 | 0.682 | 0.97% | 59.2 |
| 10x8 | llama3-405b | ring all_to_all | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | ring all_to_all, batch=8 | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather, batch=8 | INFEASIBLE | - | - | - |
| 100x8 | llama3-8b | ring all_to_all | tp=5 pp=10 cp=16 dp=1 zero=0 | 0.009 | 2.99% | 2.7 |
| 100x8 | llama3-8b | default all_gather | tp=1 pp=16 cp=1 dp=50 zero=2 | 0.011 | 4.10% | 5.3 |
| 100x8 | llama3-8b | ring all_to_all, batch=8 | tp=1 pp=25 cp=32 dp=1 zero=0 | 0.019 | 0.87% | 5.4 |
| 100x8 | llama3-8b | default all_gather, batch=8 | tp=1 pp=32 cp=25 dp=1 zero=0 | 0.027 | 0.38% | 4.3 |
| 100x8 | llama3-70b | ring all_to_all | tp=1 pp=50 cp=16 dp=1 zero=0 | 0.038 | 1.11% | 23.2 |
| 100x8 | llama3-70b | default all_gather | tp=1 pp=32 cp=1 dp=25 zero=2 | 0.048 | 0.26% | 12.8 |
| 100x8 | llama3-70b | ring all_to_all, batch=8 | tp=1 pp=40 cp=20 dp=1 zero=0 | 0.099 | 0.02% | 28.8 |
| 100x8 | llama3-70b | default all_gather, batch=8 | tp=1 pp=50 cp=16 dp=1 zero=0 | 0.124 | 0.22% | 23.2 |
| 100x8 | llama3-405b | ring all_to_all | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.128 | 7.21% | 66.3 |
| 100x8 | llama3-405b | default all_gather | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.170 | 2.58% | 66.3 |
| 100x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=50 cp=8 dp=2 zero=2 | 0.503 | 2.97% | 75.0 |
| 100x8 | llama3-405b | default all_gather, batch=8 | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.560 | 2.93% | 66.3 |
| 1000x8 | llama3-8b | ring all_to_all | tp=5 pp=20 cp=16 dp=5 zero=2 | 0.009 | 1.99% | 0.5 |
| 1000x8 | llama3-8b | default all_gather | tp=1 pp=16 cp=1 dp=500 zero=2 | 0.010 | 4.48% | 5.1 |
| 1000x8 | llama3-8b | ring all_to_all, batch=8 | tp=1 pp=25 cp=20 dp=16 zero=2 | 0.015 | 0.47% | 1.2 |
| 1000x8 | llama3-8b | default all_gather, batch=8 | tp=1 pp=25 cp=16 dp=20 zero=2 | 0.022 | 0.67% | 1.2 |
| 1000x8 | llama3-70b | ring all_to_all | tp=8 pp=25 cp=20 dp=2 zero=2 | 0.027 | 5.09% | 3.3 |
| 1000x8 | llama3-70b | default all_gather | tp=8 pp=25 cp=20 dp=2 zero=2 | 0.040 | 3.16% | 3.3 |
| 1000x8 | llama3-70b | ring all_to_all, batch=8 | tp=5 pp=20 cp=80 dp=1 zero=0 | 0.056 | 0.03% | 11.4 |
| 1000x8 | llama3-70b | default all_gather, batch=8 | tp=8 pp=20 cp=50 dp=1 zero=0 | 0.079 | 0.33% | 7.1 |
| 1000x8 | llama3-405b | ring all_to_all | tp=8 pp=40 cp=25 dp=1 zero=0 | 0.062 | 1.04% | 20.4 |
| 1000x8 | llama3-405b | default all_gather | tp=8 pp=40 cp=25 dp=1 zero=0 | 0.091 | 0.61% | 20.4 |
| 1000x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=100 cp=80 dp=1 zero=0 | 0.149 | 2.15% | 65.6 |
| 1000x8 | llama3-405b | default all_gather, batch=8 | tp=2 pp=80 cp=50 dp=1 zero=0 | 0.216 | 0.96% | 40.9 |
