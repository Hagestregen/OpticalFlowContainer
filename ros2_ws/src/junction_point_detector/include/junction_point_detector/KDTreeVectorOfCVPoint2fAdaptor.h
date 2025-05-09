#pragma once

#include "junction_point_detector/vendor/nanoflann/nanoflann.hpp"
#include <vector>
#include <opencv2/core/types.hpp> // Ensure you have included the opencv2 core types header

template <
    class VectorOfVectorsType, typename num_t = double, int DIM = 2,
    class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
    using self_t = KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance, IndexType>;
    using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>;

    index_t* index = nullptr; // The kd-tree index

    const VectorOfVectorsType& m_data; // Reference to the data points

    // Constructor
    KDTreeVectorOfVectorsAdaptor(
        const size_t /* dimensionality */, const VectorOfVectorsType& mat,
        const int leaf_max_size = 10, const unsigned int n_thread_build = 1)
        : m_data(mat)
    {
        assert(mat.size() != 0);
        index = new index_t(
            DIM, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_thread_build));
    }

    ~KDTreeVectorOfVectorsAdaptor() { delete index; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return m_data.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return m_data[idx].x;
        else if (dim == 1) return m_data[idx].y;
        else throw std::runtime_error("Invalid dimensionality for access.");
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

}; // end of KDTreeVectorOfVectorsAdaptor
