| fabric | model | CP convention / batch | best config | step (s) | margin | mem/GPU |
|---|---|---|---|---|---|---|
| 10x8 | llama3-8b | ring all_to_all | tp=1 pp=20 cp=4 dp=1 zero=0 | 0.017 | 1.03% | 7.7 |
| 10x8 | llama3-8b | default all_gather | tp=1 pp=16 cp=1 dp=5 zero=2 | 0.019 | 2.51% | 6.6 |
| 10x8 | llama3-8b | ring all_to_all, batch=8 | tp=5 pp=1 cp=16 dp=1 zero=0 | 0.085 | 0.34% | 27.0 |
| 10x8 | llama3-8b | default all_gather, batch=8 | tp=1 pp=16 cp=5 dp=1 zero=0 | 0.097 | 1.03% | 9.2 |
| 10x8 | llama3-70b | ring all_to_all | tp=1 pp=40 cp=2 dp=1 zero=0 | 0.111 | 6.10% | 34.8 |
| 10x8 | llama3-70b | default all_gather | tp=1 pp=40 cp=1 dp=2 zero=2 | 0.120 | 1.42% | 23.9 |
| 10x8 | llama3-70b | ring all_to_all, batch=8 | tp=1 pp=20 cp=4 dp=1 zero=0 | 0.718 | 4.10% | 65.9 |
| 10x8 | llama3-70b | default all_gather, batch=8 | tp=1 pp=20 cp=4 dp=1 zero=0 | 0.749 | 1.60% | 65.9 |
| 10x8 | llama3-405b | ring all_to_all | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | ring all_to_all, batch=8 | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather, batch=8 | INFEASIBLE | - | - | - |
| 100x8 | llama3-8b | ring all_to_all | tp=5 pp=10 cp=16 dp=1 zero=0 | 0.009 | 3.11% | 2.8 |
| 100x8 | llama3-8b | default all_gather | tp=1 pp=16 cp=1 dp=50 zero=2 | 0.011 | 4.90% | 5.3 |
| 100x8 | llama3-8b | ring all_to_all, batch=8 | tp=1 pp=25 cp=32 dp=1 zero=0 | 0.019 | 0.98% | 5.6 |
| 100x8 | llama3-8b | default all_gather, batch=8 | tp=1 pp=32 cp=25 dp=1 zero=0 | 0.027 | 0.69% | 4.4 |
| 100x8 | llama3-70b | ring all_to_all | tp=1 pp=50 cp=16 dp=1 zero=0 | 0.040 | 1.04% | 25.8 |
| 100x8 | llama3-70b | default all_gather | tp=1 pp=32 cp=1 dp=25 zero=2 | 0.051 | 1.52% | 13.4 |
| 100x8 | llama3-70b | ring all_to_all, batch=8 | tp=1 pp=40 cp=20 dp=1 zero=0 | 0.107 | 0.45% | 32.1 |
| 100x8 | llama3-70b | default all_gather, batch=8 | tp=1 pp=50 cp=16 dp=1 zero=0 | 0.133 | 0.76% | 25.8 |
| 100x8 | llama3-405b | ring all_to_all | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.138 | 6.66% | 76.1 |
| 100x8 | llama3-405b | default all_gather | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.185 | 2.75% | 76.1 |
| 100x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=50 cp=8 dp=2 zero=3 | 0.563 | 0.85% | 76.7 |
| 100x8 | llama3-405b | default all_gather, batch=8 | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.615 | 5.09% | 76.1 |
| 1000x8 | llama3-8b | ring all_to_all | tp=5 pp=20 cp=16 dp=5 zero=2 | 0.009 | 2.13% | 0.5 |
| 1000x8 | llama3-8b | default all_gather | tp=1 pp=16 cp=1 dp=500 zero=2 | 0.010 | 5.35% | 5.2 |
| 1000x8 | llama3-8b | ring all_to_all, batch=8 | tp=1 pp=25 cp=20 dp=16 zero=2 | 0.015 | 0.45% | 1.3 |
| 1000x8 | llama3-8b | default all_gather, batch=8 | tp=1 pp=25 cp=16 dp=20 zero=2 | 0.022 | 0.67% | 1.2 |
| 1000x8 | llama3-70b | ring all_to_all | tp=8 pp=25 cp=20 dp=2 zero=2 | 0.028 | 5.48% | 3.6 |
| 1000x8 | llama3-70b | default all_gather | tp=8 pp=25 cp=20 dp=2 zero=2 | 0.041 | 2.95% | 3.6 |
| 1000x8 | llama3-70b | ring all_to_all, batch=8 | tp=5 pp=25 cp=64 dp=1 zero=0 | 0.057 | 0.20% | 10.2 |
| 1000x8 | llama3-70b | default all_gather, batch=8 | tp=8 pp=20 cp=50 dp=1 zero=0 | 0.081 | 0.88% | 7.9 |
| 1000x8 | llama3-405b | ring all_to_all | tp=8 pp=40 cp=25 dp=1 zero=0 | 0.064 | 2.59% | 23.5 |
| 1000x8 | llama3-405b | default all_gather | tp=8 pp=50 cp=20 dp=1 zero=0 | 0.094 | 0.13% | 18.8 |
| 1000x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=100 cp=80 dp=1 zero=0 | 0.159 | 2.01% | 75.4 |
| 1000x8 | llama3-405b | default all_gather, batch=8 | tp=2 pp=80 cp=50 dp=1 zero=0 | 0.227 | 0.91% | 47.1 |
