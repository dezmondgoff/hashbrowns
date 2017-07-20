# hashbrowns

A Python module for locality senstive hashing based on <a href="https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwjKx4W6spjVAhVLxVQKHZgmCqIQFggvMAI&url=http%3A%2F%2Feduardovalle.com%2Fwordpress%2Fwp-content%2Fuploads%2F2014%2F10%2Fsilva14sisapLargeScaleMetricLSH.pdf&usg=AFQjCNEzBzXg_F6lu0VWZ2sQI3x9lltrQQ">Silva et al (2014)</a>, and also <a href="https://graphics.stanford.edu/courses/cs468-06-fall/Papers/12%20lsh04.pdf">Datar et al (2004)</a>.

Still very much a work-in-progress. 

TODO:
* Add hdf5 data/storage via h5py
* Add sql data via sqlite3
* Add support for data supplied as bytes (memory/mmap), ie sequence data
* Switch to using `randomkit` from `numpy` instead of `rand` from `stdlib`
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
