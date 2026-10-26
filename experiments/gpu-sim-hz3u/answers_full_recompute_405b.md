| fabric | model | CP convention / batch | best config | step (s) | margin | mem/GPU |
|---|---|---|---|---|---|---|
| 10x8 | llama3-405b | ring all_to_all | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | ring all_to_all, batch=8 | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather, batch=8 | INFEASIBLE | - | - | - |
| 100x8 | llama3-405b | ring all_to_all | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.150 | 6.13% | 75.7 |
| 100x8 | llama3-405b | default all_gather | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.197 | 2.58% | 75.7 |
| 100x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=50 cp=8 dp=2 zero=3 | 0.656 | 0.75% | 75.9 |
| 100x8 | llama3-405b | default all_gather, batch=8 | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.708 | 4.40% | 75.7 |
| 1000x8 | llama3-405b | ring all_to_all | tp=8 pp=40 cp=25 dp=1 zero=0 | 0.065 | 2.54% | 23.5 |
| 1000x8 | llama3-405b | default all_gather | tp=8 pp=50 cp=20 dp=1 zero=0 | 0.096 | 0.13% | 18.8 |
| 1000x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=100 cp=80 dp=1 zero=0 | 0.168 | 1.89% | 75.3 |
| 1000x8 | llama3-405b | default all_gather, batch=8 | tp=2 pp=80 cp=50 dp=1 zero=0 | 0.236 | 0.86% | 47.0 |
