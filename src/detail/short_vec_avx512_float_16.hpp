/**
 * Copyright 2015 Kurt Kanzenbach
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_FLOAT_16_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_AVX512_FLOAT_16_HPP

#ifdef __AVX512__

#include <immintrin.h>
#include <libflatarray/detail/sqrt_reference.hpp>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<typename CARGO, int ARITY>
class sqrt_reference;

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

template<>
class short_vec<float, 16>
{
public:
    static const int ARITY = 16;

    typedef short_vec_strategy::avx512 strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 16>& vec);

    inline
    short_vec(const float data = 0) :
        val1(_mm512_broadcast_ss(&data))
    {}

    inline
    short_vec(const float *data) :
        val1(_mm512_loadu_ps(data))
    {}

    inline
    short_vec(const __m512& val1) :
        val1(val1)
    {}

    inline
    short_vec(const sqrt_reference<float, 16> other);

    inline
    void operator-=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_sub_ps(val1, other.val1);
    }

    inline
    short_vec<float, 16> operator-(const short_vec<float, 16>& other) const
    {
        return _mm512_sub_ps(val1, other.val1);
    }

    inline
    void operator+=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_add_ps(val1, other.val1);
    }

    inline
    short_vec<float, 16> operator+(const short_vec<float, 16>& other) const
    {
        return _mm512_add_ps(val1, other.val1);
    }

    inline
    void operator*=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_mul_ps(val1, other.val1);
    }

    inline
    short_vec<float, 16> operator*(const short_vec<float, 16>& other) const
    {
        return _mm512_mul_ps(val1, other.val1);
    }

    inline
    void operator/=(const short_vec<float, 16>& other)
    {
        val1 = _mm512_mul_ps(val1, _mm512_rcp_ps(other.val1));
    }

    inline
    void operator/=(const sqrt_reference<float, 16>& other);

    inline
    short_vec<float, 16> operator/(const short_vec<float, 16>& other) const
    {
        return _mm512_mul_ps(val1, _mm512_rcp_ps(other.val1));
    }

    inline
    short_vec<float, 16> operator/(const sqrt_reference<float, 16>& other) const;

    inline
    short_vec<float, 16> sqrt() const
    {
        return _mm512_sqrt_ps(val1);
    }

    inline
    void store(float *data) const
    {
        _mm512_storeu_ps(data, val1);
    }

private:
    __m512 val1;
};

inline
void operator<<(float *data, const short_vec<float, 16>& vec)
{
    vec.store(data);
}

template<>
class sqrt_reference<float, 16>
{
public:
    template<typename OTHER_CARGO, int OTHER_ARITY>
    friend class short_vec;

    sqrt_reference(const short_vec<float, 16>& vec) :
        vec(vec)
    {}

private:
    short_vec<float, 16> vec;
};

#ifdef __ICC
#pragma warning pop
#endif

inline
short_vec<float, 16>::short_vec(const sqrt_reference<float, 16> other) :
    val1(_mm512_sqrt_ps(other.vec.val1))
{}

inline
void short_vec<float, 16>::operator/=(const sqrt_reference<float, 16>& other)
{
    val1 = _mm512_mul_ps(val1, _mm512_rsqrt_ps(other.vec.val1));
}

inline
short_vec<float, 16> short_vec<float, 16>::operator/(const sqrt_reference<float, 16>& other) const
{
    return _mm512_mul_ps(val1, _mm512_rsqrt_ps(other.vec.val1));
}

inline
sqrt_reference<float, 16> sqrt(const short_vec<float, 16>& vec)
{
    return sqrt_reference<float, 16>(vec);
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 16>& vec)
{
    const float *data1 = reinterpret_cast<const float *>(&vec.val1);
    __os << "[" << data1[0] << ", " << data1[1]  << ", " << data1[2]  << ", " << data1[3]  << ", " << data1[4]  << ", " << data1[5]  << ", " << data1[6]  << ", " << data1[7] << "]";
    return __os;
}

}

#endif
#endif

#endif
