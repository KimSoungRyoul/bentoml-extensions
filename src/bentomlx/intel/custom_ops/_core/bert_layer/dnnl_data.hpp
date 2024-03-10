// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_DATA__
#define __DNNL_DATA__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"

#include <string>

namespace dnnl_wrappers {

class DataSource {
public:
    DataSource(const dnnl::memory& mem = {}, BuildAttrs attr = {})
        : mem_{mem}
        , attr_{attr} {}

    DataSource(const DataSource& other) = default;
    DataSource(DataSource&& other) = default;
    DataSource& operator=(const DataSource& other) = default;
    DataSource& operator=(DataSource&& other) = default;
    virtual ~DataSource() = default;

    virtual dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) {
        if (!mem_) {
             return mem_;
        }

        if (attr_.Empty() && mem_.get_engine() == stm.get_engine() && mem_.get_desc() == md) {
            return mem_;
        }
        dnnl::memory result{md, stm.get_engine()};

        // No need to check for nullptr, implicitly convert to dnnl::primitive_attr
        dnnl::reorder rdr{mem_, result, attr_.GetAttrs()};
        auto args = attr_.GetArgs();
        args.insert({
            {DNNL_ARG_SRC, mem_},
            {DNNL_ARG_DST, result}});
        rdr.execute(stm, args);
        return result;
    }

protected:
    dnnl::memory mem_;
    BuildAttrs attr_;
};

class CachedDataSource : public DataSource {
public:
    using DataSource::DataSource;

    dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) override {
        if (!cached_mem_ || cached_mem_.get_engine() != stm.get_engine() || cached_mem_.get_desc() != md) {
            cached_mem_ = DataSource::GetData(stm, md);
        }
        return cached_mem_;
    }

private:
    dnnl::memory cached_mem_;
};

/// @brief  Optimized DataSource which does not modify original memory
class ImmutableDataSource : public DataSource {
public:
    ImmutableDataSource(const dnnl::memory& mem = {}) : DataSource(mem, {}) {}
    dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) override {
        // unused in Release build:
        (void)(stm);
        (void)(md);

        if (!mem_) {
             return mem_;
        }

        assert(attr_.Empty() && mem_.get_engine() == stm.get_engine() && mem_.get_desc() == md);
        return mem_;
    }
};

inline dnnl::memory::format_tag PlainFormatTag(size_t ndims, bool trans = false) {
    using ft = dnnl::memory::format_tag;
    switch (ndims) {
        case 1: return ft::a;
        case 2: return trans ? ft::ba : ft::ab;
        case 3: return trans ? ft::acb : ft::abc;
        case 4: return trans ? ft::abdc : ft::abcd;
        default: return ft::undef;
    }
}

template <class T>
dnnl::memory AttachMemory(const dnnl::engine& eng, dnnl::memory::dims dims, T* data, bool trans = false) {
    const auto dt = DnnlDataType<T>::value;
    dnnl::memory::desc md{dims, dt, PlainFormatTag(dims.size(), trans)};
    return dnnl::memory{md, eng, data};
}

template <class T>
dnnl::memory CloneMemory(const dnnl::engine& eng, dnnl::stream& stm, dnnl::memory::dims dims, const T* data, bool trans = false) {
    const auto dt = DnnlDataType<T>::value;
    auto src = AttachMemory(eng, dims, const_cast<T*>(data), trans);
    dnnl::memory::desc md{dims, dt, dnnl::memory::dims{}};
    dnnl::memory dst{md, stm.get_engine()};
    dnnl::reorder{src, dst}.execute(stm, src, dst);
    stm.wait();
    return dst;
}

template <class T, class A = std::allocator<T>>
dnnl::memory ToMemory(const dnnl::engine& eng, dnnl::stream& stm, const std::vector<T,A> v) {
    dnnl::memory::dim size = v.size();
    return CloneMemory(eng, stm, {static_cast<dnnl::memory::dim>(size)}, v.data());
}

template <class T>
dnnl::memory ToMemory(const dnnl::engine& eng, dnnl::stream& stm, const T v) {
    return CloneMemory(eng, stm, {1}, &v);
}

/**
 * @brief Reshape a memory object with a validity check.
 *
 * @param memory The memory object to reshape.
 * @param dims The new dimensions.
 * @return The reshaped memory.
 * @throws dnnl::error if the reshape cannot be performed.
 */
dnnl::memory ReshapeMemory(const dnnl::memory& memory, const dnnl::memory::dims& dims)
{
    dnnl::memory::desc md = memory.get_desc().reshape(dims);
    return dnnl::memory{md, memory.get_engine(), memory.get_data_handle()};
}

/**
 * @brief Reinterpret a dnnl::memory with different dimensions and format, but the same data type. Does NOT perform
 * any validity checks.
 *
 * @param mem The memory object to reinterpret.
 * @param layout The target descriptor, data type will be overwritten.
 * @return The reinterpreted memory.
 */
dnnl::memory ReLayoutMemory(const dnnl::memory& mem, const dnnl::memory::desc& layout) {
    assert(layout.get_data_type() == mem.get_desc().get_data_type());
    assert(layout.get_size() <= mem.get_desc().get_size());
    return dnnl::memory{layout, mem.get_engine(), mem.get_data_handle()};
}

DataSource ScaledData(const dnnl::memory& mem, const dnnl::memory& scale, const dnnl::memory& zero_point = {}) {
    return DataSource(mem, BuildAttrs().Scale(scale, DNNL_ARG_DST).ZeroPoint(zero_point, DNNL_ARG_DST));
}

CachedDataSource ScaledCachedData(const dnnl::memory& mem, const dnnl::memory& scale, const dnnl::memory& zero_point = {}) {
    return CachedDataSource(mem, BuildAttrs().Scale(scale, DNNL_ARG_DST).ZeroPoint(zero_point, DNNL_ARG_DST));
}

} // namespace dnnl_wrappers

#endif //__DNNL_DATA__
