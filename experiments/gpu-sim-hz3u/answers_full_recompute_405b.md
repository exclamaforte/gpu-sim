| fabric | model | CP convention / batch | best config | step (s) | margin | mem/GPU |
|---|---|---|---|---|---|---|
| 10x8 | llama3-405b | ring all_to_all | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | ring all_to_all, batch=8 | INFEASIBLE | - | - | - |
| 10x8 | llama3-405b | default all_gather, batch=8 | INFEASIBLE | - | - | - |
| 100x8 | llama3-405b | ring all_to_all | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.154 | 5.97% | 75.7 |
| 100x8 | llama3-405b | default all_gather | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.201 | 2.52% | 75.7 |
| 100x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=50 cp=8 dp=2 zero=3 | 0.687 | 0.72% | 75.7 |
| 100x8 | llama3-405b | default all_gather, batch=8 | tp=1 pp=100 cp=8 dp=1 zero=0 | 0.739 | 4.20% | 75.7 |
| 1000x8 | llama3-405b | ring all_to_all | tp=8 pp=40 cp=25 dp=1 zero=0 | 0.066 | 2.53% | 23.4 |
| 1000x8 | llama3-405b | default all_gather | tp=8 pp=50 cp=20 dp=1 zero=0 | 0.096 | 0.13% | 18.8 |
| 1000x8 | llama3-405b | ring all_to_all, batch=8 | tp=1 pp=100 cp=80 dp=1 zero=0 | 0.171 | 1.85% | 75.3 |
| 1000x8 | llama3-405b | default all_gather, batch=8 | tp=2 pp=80 cp=50 dp=1 zero=0 | 0.239 | 0.85% | 47.0 |
