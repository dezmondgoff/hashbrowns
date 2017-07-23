TODO:
* Add hdf5 data/storage via h5py
* Add sql data via sqlite3
* Add support for data supplied as bytes (memory/mmap), ie sequence data
* ~~Switch to using `randomkit` from `numpy` instead of `rand` from `stdlib`~~
* Write all the tests! Test all the things!
  * Test all data and storage structures
  * Test alignment distance extension functions for correctness
* Add kmedoids/kmeans algorithm(s) (not extremely necessary, but nice for completeness)
* Demos
  * lsh for denstity-based clustering (as a fast approximation for kNN problem in OPTICS algorithm)
  * lsh accelerated multiple alignment
  * lsh for classification
  * lsh for estimating mutual information
  * pretty much scaling any kNN based algorithm to large data space
