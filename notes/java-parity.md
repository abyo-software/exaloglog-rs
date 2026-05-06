# Java reference parity

The fixtures in `tests/fixtures/d*_p*_n*.hex` were captured from
Dynatrace's official ExaLogLog reference implementation at
[github.com/dynatrace-research/exaloglog-paper](https://github.com/dynatrace-research/exaloglog-paper).
This Rust crate's register state matches that reference bit-for-bit
across every captured configuration — `t = 2`, `d ∈ {20, 24}`,
`p ∈ {4, 8, 12}`, `n ∈ {100, 1000, 10000}` (18 total).

## Capture procedure

The fixtures were produced by compiling the Java reference's
`ExaLogLog`, `DistinctCountUtil`, `MartingaleEstimator`, and
`MLBiasCorrectionConstants` classes against `hash4j 0.20.0`, building an
`ExaLogLog(t, d, p)`, inserting `splitmix64(0..n)`, and printing
`getState()` as lowercase hex.

The same Rust splitmix64 is used in `tests/java_parity.rs`. The constants
match exactly — both implementations use the original Stafford-mixed
splitmix64.

## Regenerating

If you bump the reference version of the Java implementation or want
new (p, n) coverage, regenerate the fixtures with the following steps:

```sh
# 1. Get the Java reference and hash4j.
git clone --depth 1 https://github.com/dynatrace-research/exaloglog-paper /tmp/exaloglog-paper
mkdir -p /tmp/exa-libs
curl -sSL -o /tmp/exa-libs/hash4j-0.20.0.jar \
  https://repo1.maven.org/maven2/com/dynatrace/hash4j/hash4j/0.20.0/hash4j-0.20.0.jar

# 2. Compile ExaLogLog and its dependencies (skip the test sources).
cd /tmp/exaloglog-paper/java/src/main/java
find . -name 'ExaLogLog.java' -o -name 'DistinctCountUtil.java' \
       -o -name 'MartingaleEstimator.java' \
       -o -name 'MLBiasCorrectionConstants.java' \
  | xargs javac -cp /tmp/exa-libs/hash4j-0.20.0.jar -d /tmp/exa-classes

# 3. Build the small Main.java driver (see notes/java-parity-main.txt
#    for the exact source).
cd /tmp
javac -cp /tmp/exa-classes:/tmp/exa-libs/hash4j-0.20.0.jar Main.java -d /tmp/exa-classes

# 4. Generate fixtures for the configurations you want.
for d in 24 20; do
  for p in 4 8 12; do
    for n in 100 1000 10000; do
      java -cp /tmp/exa-classes:/tmp/exa-libs/hash4j-0.20.0.jar Main 2 $d $p $n \
        > tests/fixtures/d${d}_p${p}_n${n}.hex
    done
  done
done

# 5. Run the parity tests.
cargo test --test java_parity
```

The `Main.java` source is in `notes/java-parity-main.txt`.
