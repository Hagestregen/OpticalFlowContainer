#pragma once

#include "vendor/nanoflann/nanoflann.hpp"
#include <vector>
#include <opencv2/core/types.hpp> // Ensure you have included the opencv2 core types header

template <
    class VectorOfVectorsType, typename num_t = double,
    class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeKeyPointsAdaptor
{
    using self_t = KDTreeKeyPointsAdaptor<VectorOfVectorsType, num_t, Distance, IndexType>;
    using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, 2, IndexType>;

    index_t* index = nullptr; // The kd-tree index

    const VectorOfVectorsType& m_data; // Reference to the data points

    // Constructor
    KDTreeKeyPointsAdaptor(
        const size_t /* dimensionality */, const VectorOfVectorsType& mat,
        const int leaf_max_size = 10, const unsigned int n_thread_build = 1)
        : m_data(mat)
    {
        assert(mat.size() != 0);
        index = new index_t(
            2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_thread_build));
    }

    ~KDTreeKeyPointsAdaptor() { delete index; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return m_data.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return m_data[idx].pt.x;
        else if (dim == 1) return m_data[idx].pt.y;
        else throw std::runtime_error("Invalid dimensionality for access.");
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

}; // end of KDTreeKeyPointsAdaptor
