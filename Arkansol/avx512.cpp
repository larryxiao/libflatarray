#include "stdio.h"
#include "immintrin.h"
//#include "zmmintrin.h" not providedm immintrin.h includes avx512* headers
//avx512cdintrin.h
//avx512erintrin.h
//avx512fintrin.h
//avx512pfintrin.h
//avxintrin.h

//# compile
//../Software/gcc_install_5.0.0-mpx-r214719/bin/x86_64-pc-linux-gnu-c++ -mavx512f -B ../Software/binutils-gdb_install_2.24.51.20140422/bin -c avx512.cpp -o avx512.o
//# link
//g++ -L /usr/lib/x86_64-linux-gnu/ avx512.o
//# running
//../Software/sde-external-7.21.0-2015-04-01-lin/sde -mpx-mode -- ./a.out

int main(int argc, const char *argv[])
{
    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4] = {8.0, 7.0, 6.0, 5.0};
    double e[4] = {0.0};
    __m256d test0 = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    __m256d val0 = _mm256_load_pd(a);
    __m256d val1 = _mm256_load_pd(b);
    __m256d val2 = _mm256_add_pd(val0, val1);
    _mm256_store_pd(e, val2); 
    printf("avx256 %f %f %f %f\n", e[0], e[1], e[2], e[3]);

    double c[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double d[8] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double f[8] = {0.0};
    __m512d test1 = _mm512_set_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    __m512d val3 = _mm512_load_pd(c);
    __m512d val4 = _mm512_load_pd(d);
    __m512d val5 = _mm512_add_pd(val3, val4);
    _mm512_store_pd(f, val5); 
    printf("avx512 %f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3],
            f[4], f[5], f[6], f[7]);

    printf("test\n");
    return 0;
}
