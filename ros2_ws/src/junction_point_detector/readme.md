


The DBSCAN is from https://github.com/Eleobert/dbscan.
It uses a newer version of c++ (c++20 or newer) which has a built in library called span. This does not exist for the current
used version (c++14, see CMakeLists.txt) so span.hpp is imported into tcb and used from there.