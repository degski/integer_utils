
// MIT License
//
// Copyright (c) 2018, 2019 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <intrin.h>
#include <immintrin.h>

#define _CRT_RAND_S
#include <cmath>
#include <cassert>

#include "sprp32.h" // https://github.com/wizykowski/miller-rabin
#include "sprp64.h"

#include "integer_utils.hpp"
#include "shift_rotate_avx2.hpp"


namespace iu {

void nl ( ) noexcept {
    std::putchar ( '\n' );
}

std::uint16_t mod_mul_inv ( const std::uint16_t a_ ) noexcept {
    // Given odd a, compute x such that a * x = 1 over 16 bits.
    const std::uint8_t b = ( std::uint8_t ) a_;		    // low 16 bits of a
    std::uint8_t x = ( ( ( b + 2u ) & 4u ) << 1 ) + b;	// low  4 bits of inverse
    x = ( 2u - b * x ) * x;							    // low  8 bits of inverse
    return ( 2u - a_ * x ) * x;						    // 16 bits of inverse
}

std::uint32_t mod_mul_inv ( const std::uint32_t a_ ) noexcept {
    // Given odd a, compute x such that a * x = 1 over 32 bits.
    const std::uint16_t b = ( std::uint16_t ) a_;	    // low 16 bits of a
    std::uint16_t x = ( ( ( b + 2u ) & 4u ) << 1 ) + b;	// low  4 bits of inverse
    x = ( 2u - b * x ) * x;							    // low  8 bits of inverse
    x = ( 2u - b * x ) * x;							    // low 16 bits of inverse
    return ( 2u - a_ * x ) * x;						    // 32 bits of inverse
}

std::uint64_t mod_mul_inv ( const std::uint64_t a_ ) noexcept {
    // Given odd a, compute x such that a * x = 1 over 64 bits.
    const std::uint32_t b = ( std::uint32_t ) a_;	    // low 32 bits of a
    std::uint32_t x = ( ( ( b + 2u ) & 4u ) << 1 ) + b;	// low  4 bits of inverse
    x = ( 2u - b * x ) * x;							    // low  8 bits of inverse
    x = ( 2u - b * x ) * x;							    // low 16 bits of inverse
    x = ( 2u - b * x ) * x;							    // low 32 bits of inverse
    return ( 2u - a_ * x ) * x;						    // 64 bits of inverse
}

bool is_prime ( const std::uint32_t n_ ) noexcept {
    assert ( n_ & std::uint32_t { 1 } );
    static const std::uint32_t bases [ 3 ] = { 2UL, 7UL, 61UL };
    return efficient_mr32 ( bases, 3, n_ ) == 1;
}

bool is_prime ( const std::uint64_t n_ ) noexcept {
    assert ( n_ & std::uint64_t { 1 } );
    static const std::uint64_t bases [ 7 ] = { 2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL };
    return efficient_mr64 ( bases, 7, n_ ) == 1;
}

// Random.

// Seeding.

#if defined ( __AVX2__ ) && defined ( __GNUC__ )
void seed ( std::uint32_t & s_ ) noexcept {
    _rdseed32_step ( &s_ );
}

void seed ( std::uint64_t & s_ ) noexcept {
    _rdseed64_step ( &s_ );
}
#else
#ifdef _WIN32
void seed ( std::uint32_t & s_ ) noexcept {
    rand_s ( &s_ );
}

void seed ( std::uint64_t & s_ ) noexcept {
    rand_s ( ( ( std::uint32_t * ) & s_ ) + 0 );
    rand_s ( ( ( std::uint32_t * ) & s_ ) + 1 );
}
#else
}
#include <random>
namespace iu {
void seed ( std::uint32_t & s_ ) noexcept {
    s_ = [ ] ( ) { std::random_device rdev; return ( std::uint32_t ) rdev ( ); } ( );
}

void seed ( std::uint64_t & s_ ) noexcept {
    if constexpr ( std::is_same<typename std::random_device::result_type, std::uint64_t>::value ) {
        s_ = [ ] ( ) { std::random_device rdev; return ( std::uint32_t ) rdev ( ); } ( );
    }
    else {
        auto _seed = [ ] ( ) { std::random_device rdev; return ( std::uint64_t ) rdev ( ); };
        s_ = ( _seed ( ) << 32 ) & _seed ( );
    }
}
#endif
#endif

std::uintmax_t seed ( ) noexcept {
    std::uintmax_t r;
    iu::seed ( r );
    return r;
}


// #ifdef __AVX2__

xoroshiro4x128plusavx::xoroshiro4x128plusavx ( ) noexcept {
    m_s0 = _mm256_set_epi64x ( iu::seed ( ), iu::seed ( ), iu::seed ( ), iu::seed ( ) );
    m_s1 = _mm256_set_epi64x ( iu::seed ( ), iu::seed ( ), iu::seed ( ), iu::seed ( ) );
    m_i  = start_case ( );
}

xoroshiro4x128plusavx::xoroshiro4x128plusavx ( const std::uint64_t s_ ) noexcept {
    seed ( s_ );
}

void xoroshiro4x128plusavx::seed ( const std::uint64_t s_ ) noexcept {
    sax::splitmix64 rng ( s_ );
    m_s0 = _mm256_set_epi64x ( rng ( ), rng ( ), rng ( ), rng ( ) );
    m_s1 = _mm256_set_epi64x ( rng ( ), rng ( ), rng ( ), rng ( ) );
    m_i  = start_case ( );
}

typename xoroshiro4x128plusavx::result_type xoroshiro4x128plusavx::operator ( ) ( ) noexcept {
    switch ( m_i ) {
        case 0:  m_i = 1; return _mm256_extract_epi64 ( m_r, 0 );
        case 1:  m_i = 2; return _mm256_extract_epi64 ( m_r, 1 );
        case 2:  m_i = 3; return _mm256_extract_epi64 ( m_r, 2 );
        default:
        {
            m_i  = 0;
            m_r  = _mm256_add_epi64 ( m_s0, m_s1 );

            m_s1 = _mm256_xor_si256 ( m_s1, m_s0 );
            m_s0 = _mm256_xor_si256 ( _mm256_xor_si256 ( _mm256_rli_si256 ( m_s0, 24 ), m_s1 ), _mm256_slli_epi64 ( m_s1, 16 ) );
            m_s1 = _mm256_rli_si256 ( m_s1, 37 );

            return _mm256_extract_epi64 ( m_r, 3 );
        }
    }
}

void print_bits ( __m256i n ) noexcept { // little-endian
    print_bits ( ( std::uint64_t ) _mm256_extract_epi64 ( n, 3 ) );
    print_bits ( ( std::uint64_t ) _mm256_extract_epi64 ( n, 2 ) );
    print_bits ( ( std::uint64_t ) _mm256_extract_epi64 ( n, 1 ) );
    print_bits ( ( std::uint64_t ) _mm256_extract_epi64 ( n, 0 ) );
}

void print_u64 ( __m256i n ) noexcept {
    printf ( "%I64u %I64u %I64u %I64u", ( std::uint64_t ) _mm256_extract_epi64 ( n, 3 ), ( std::uint64_t ) _mm256_extract_epi64 ( n, 2 ), ( std::uint64_t ) _mm256_extract_epi64 ( n, 1 ), ( std::uint64_t ) _mm256_extract_epi64 ( n, 0 ) );
}

// #endif
}
/*

#include <stdio.h>

int main435345 ( ) {

    __int128_t test = 10;
    while ( test>0 ) {
        int myTest = ( int ) test;
        printf ( "? %d\n", myTest );
        test--;
    }

}




__uint128_t FASTMUL128 ( const __uint128_t TA, const __uint128_t TB ) {
    union {
        __uint128_t WHOLE;
        struct {
            unsigned long long int LWORDS [ 2 ];
        } SPLIT;
    } KEY;
    register unsigned long long int __RAX, __RDX, __RSI, __RDI;
    __uint128_t RESULT;

    KEY.WHOLE = TA;
    __RAX = KEY.SPLIT.LWORDS [ 0 ];
    __RDX = KEY.SPLIT.LWORDS [ 1 ];
    KEY.WHOLE = TB;
    __RSI = KEY.SPLIT.LWORDS [ 0 ];
    __RDI = KEY.SPLIT.LWORDS [ 1 ];
    __asm__ __volatile__ (
        "movq           %0,                             %%rax                   \n\t"
        "movq           %1,                             %%rdx                   \n\t"
        "movq           %2,                             %%rsi                   \n\t"
        "movq           %3,                             %%rdi                   \n\t"
        "movq           %%rsi,                          %%rbx                   \n\t"
        "movq           %%rdi,                          %%rcx                   \n\t"
        "movq           %%rax,                          %%rsi                   \n\t"
        "movq           %%rdx,                          %%rdi                   \n\t"
        "xorq           %%rax,                          %%rax                   \n\t"
        "xorq           %%rdx,                          %%rdx                   \n\t"
        "movq           %%rdi,                          %%rax                   \n\t"
        "mulq           %%rbx                                                   \n\t"
        "xchgq          %%rbx,                          %%rax                   \n\t"
        "mulq           %%rsi                                                   \n\t"
        "xchgq          %%rax,                          %%rsi                   \n\t"
        "addq           %%rdx,                          %%rbx                   \n\t"
        "mulq           %%rcx                                                   \n\t"
        "addq           %%rax,                          %%rbx                   \n\t"
        "movq           %%rsi,                          %%rax                   \n\t"
        "movq           %%rbx,                          %%rdx                   \n\t"
        "movq           %%rax,                          %0                      \n\t"
        "movq           %%rdx,                          %1                      \n\t"
        "movq           %%rsi,                          %2                      \n\t"
        "movq           %%rdi,                          %3                      \n\t"
        : "=m"( __RAX ), "=m"( __RDX ), "=m"( __RSI ), "=m"( __RDI )
        : "m"( __RAX ), "m"( __RDX ), "m"( __RSI ), "m"( __RDI )
        : "rax", "rbx", "ecx", "rdx", "rsi", "rdi"
    );
    KEY.SPLIT.LWORDS [ 0 ] = __RAX;
    KEY.SPLIT.LWORDS [ 1 ] = __RDX;
    RESULT = KEY.WHOLE;
    return RESULT;
}


{
uint64_t hi, lo;
// hi,lo = 64bit x 64bit multiply of c[0] and b[0]

__asm__ ( "mulq %3\n\t"
    : "=d" ( hi ),
    "=a" ( lo )
    : "%a" ( c [ 0 ] ),
    "rm" ( b [ 0 ] )
    : "cc" );

a [ 0 ] += hi;
a [ 1 ] += lo;
 }

 */