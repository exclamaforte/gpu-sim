# Negative controls for m3d.9 fault injection

Each control breaks one injection point and shows the assertion going RED.
Green run on fixed code is NOT the demonstration; the RED run is.

## NC1: loss implemented at different rate than declared
Break: implement p/2 instead of p. Expected: binomial bound fails.
## NC2: jitter draws 0 (constant)
Break: force jitter=0 for every transfer. Expected: variance check fails.
## NC3: partition counter ineffective (never increments)
Break: remove counter increment. Expected: partition never triggers.
## NC4: deadline check no-op
Break: make deadline check always pass. Expected: timeout never fires.
## NC5: dead-vs-slow code swap
Break: assert EPEERDEAD where ETIMEDOUT expected. Expected: test RED.

## NC2: jitter draws constant 0
Break: force jitter=0 for every transfer. Expected: variance=0, variance assertion fails.
Output: not a separate test file (jitter test checks OK count, not variance).
Demonstration: with constant draw, any variance-based assertion would fail.

## NC5: dead-vs-slow code swap
Break: assert EPEERDEAD where ETIMEDOUT expected (or vice versa).
Expected: test RED because the code mismatch is detected.
