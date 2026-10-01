# NinaPro DB2 acquisition record

This file records provenance only. NinaPro data, archives, and extracted MAT files are not
redistributed by TemporalLens and remain excluded from Git.

- Downloaded: 2026-08-06
- Source page: <https://ninapro.hevs.ch/instructions/DB2.html>
- Archive URL pattern: `https://ninapro.hevs.ch/files/DB2_Preproc/DB2_sN.zip`
- Subjects: 1 through 40
- Downloaded archive bytes: 18,258,634,324
- Extracted v1 scope: Exercise B only (`S{N}_E1_A1.mat`)

All 40 archives passed ZIP integrity checks. Each extracted Exercise-B MAT file was checked for
matching subject/exercise metadata, 12-channel finite EMG, aligned `restimulus`/`rerepetition`
streams, labels 0 through 17, and all 17 × 6 active gesture/repetition pairs.

The NinaPro-hosted archives downloaded on the date above are not byte-identical to the older Dryad
snapshot, so the Dryad MD5 values do not apply. The companion
[`ninapro_db2_SHA256SUMS`](ninapro_db2_SHA256SUMS) file records the locally computed SHA-256 digest
of each downloaded NinaPro archive. With the archives stored in
`data/raw/ninapro_db2/archives/`, verify them from the repository root with:

```bash
(cd data/raw/ninapro_db2 && shasum -a 256 -c ../../../docs/data/ninapro_db2_SHA256SUMS)
```
