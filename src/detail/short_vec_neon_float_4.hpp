#ifndef FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_4_HPP
#define FLAT_ARRAY_DETAIL_SHORT_VEC_NEON_FLOAT_4_HPP

#ifdef __NEON__
#include <arm_neon.h>

#ifndef __CUDA_ARCH__

namespace LibFlatArray {

template<typename CARGO, int ARITY>
class short_vec;

template<>
class short_vec<float, 4>
{
public:
    static const int ARITY = 4;

    typedef short_vec_strategy::neon strategy;

    template<typename _CharT, typename _Traits>
    friend std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& __os,
        const short_vec<float, 4>& vec);

    inline
    short_vec(const float data = 0) :
        val1(vdupq_n_f32(data))
    {}

    inline
    short_vec(const float *data) :
        val1(vld1q_f32( *(data +  0) )),
    {}

    inline
    short_vec(const float32x4_t& val1) :
        val1(val1)
    {}

    inline
    void operator-=(const short_vec<float, 4>& other)
    {
        val1 = vsubq_f32(val1, other);
    }

    inline
    short_vec<float, 4> operator-(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(vsubq_f32(val1, other));
    }

    inline
    void operator+=(const short_vec<float, 4>& other)
    {
    	val1 = vaddq_f32(val1, other);
    }

    inline
    short_vec<float, 4> operator+(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(val1 = vaddq_f32(val1, other));
    }

    inline
    void operator*=(const short_vec<float, 4>& other)
    {
    	val1 = vmulq_f32(val1, other);
    }

    inline
    short_vec<float, 4> operator*(const short_vec<float, 4>& other) const
    {
        return short_vec<float, 4>(val1 = vmulq_f32(val1, other));
    }

    inline
    void operator/=(const short_vec<float, 4>& other)
    {
    	val1 = vmulq_f32(val1, vrecpeq_f32(other));
    }

    inline
    short_vec<float, 4> operator/(const short_vec<float, 4>& other) const
    {
    	return short_vec<float, 4>(val1 = vmulq_f32(val1, vrecpeq_f32(other)));
    }

    inline
    short_vec<float, 4> sqrt() const
    {
        return short_vec<float, 4>(
            val1 = vsqrtq_f32(val1);
    }

    inline
    void store(float *data) const
    {
        *(data +  0) = val1.val[0];
        *(data +  1) = val1.val[1];
        *(data +  2) = val1.val[2];
        *(data +  3) = val1.val[3];
    }

private:
    float32x4_t val1;
};

inline
void operator<<(float *data, const short_vec<float, 4>& vec)
{
    vec.store(data);
}

inline
short_vec<float, 4> sqrt(const short_vec<float, 4>& vec)
{
    return vec.sqrt();
}

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const short_vec<float, 4>& vec)
{
    __os << "["  << val1.val[0]  << ", " << val1.val[1]  << ", " << val1.val[2]  << ", " << val1.val[3]
         << "]";
    return __os;
}

}

#endif
#endif

#endif
