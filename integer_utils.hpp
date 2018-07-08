
// MIT License
//
// Copyright (c) 2018 degski
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

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <immintrin.h>
#include <cassert>
#include <cstdint>

#include <limits>
#include <string>
#include <type_traits>


#ifdef NDEBUG
#pragma comment ( lib, "integer_utils-s.lib" )
#else
#pragma comment ( lib, "integer_utils-s-d.lib" )
#endif

namespace iu {

namespace detail {
template<typename T>
using is_string = std::is_base_of<std::basic_string<typename T::value_type>, T>;
}

// Greatest Common Denominator.
template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T gcd ( T a_, T b_ ) noexcept {
    while ( true ) {
        if ( !a_ ) {
            return b_;
        }
        b_ %= a_;
        if ( !b_ ) {
            return a_;
        }
        a_ %= b_;
    }
}

// Least Common Multiple.
template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T lcm ( const T a_, const T b_ ) noexcept {
    const T t = gcd<T> ( a_, b_ );
    return t ? a_ / t * b_ : T ( 0 );
}


// In number theory, two integers a and b are said to be relatively prime, mutually prime, or
// coprime (also spelled co-prime) if the only positive integer that divides both of them is 1.
template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr bool are_coprime ( const T a_, const T b_ ) noexcept {
    assert ( a_ > 0 && b_ > 0 );
    return gcd<T> ( a_, b_ ) == T ( 1 );
}

// Integer LogN.
template<int Base, typename T, typename sfinae = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T iLog ( const T n_, const T p_ = T ( 0 ) ) noexcept {
    return n_ < Base ? p_ : iLog<Base, T, sfinae> ( n_ / Base, p_ + 1 );
}

// Integer Log2.
template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T ilog2 ( const T n_ ) noexcept {

    return iLog<2, T> ( n_ );
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T next_power_2 ( const T n_ ) noexcept {
    return n_ > 2 ? T ( 1 ) << ( ilog2<T> ( n_ - 1 ) + 1 ) : n_;
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr bool is_power_2 ( const T n_ ) noexcept {
    return n_ && !( n_ & ( n_ - 1 ) );
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T sum2n ( const T n_ ) noexcept {
    return ( n_ * ( n_ + 1 ) ) / T ( 2 );
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T sumMToN ( const T m_, const T n_ ) noexcept {
    return ( ( n_ * ( n_ + 1 ) ) - ( m_ * ( m_ + 1 ) ) ) / T ( 2 );
}

// Pointer Alignment.
inline std::uint32_t pointer_alignment ( const void * p_ ) noexcept {
    return ( std::uint32_t ) ( ( std::uintptr_t ) p_ & ( std::uintptr_t ) - ( ( std::intptr_t ) p_ ) );
}

// Gray Coding.
template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T dec2gray ( const T i_ ) noexcept {
    return i_ ^ ( i_ >> 1 );
}

constexpr std::uint8_t gray2dec ( std::uint8_t g_ ) noexcept {
    g_ ^= g_ >> 4;
    g_ ^= g_ >> 2;
    g_ ^= g_ >> 1;
    return g_;
}

constexpr std::uint16_t gray2dec ( std::uint16_t g_ ) noexcept {
    g_ ^= g_ >> 8;
    g_ ^= g_ >> 4;
    g_ ^= g_ >> 2;
    g_ ^= g_ >> 1;
    return g_;
}

constexpr std::uint32_t gray2dec ( std::uint32_t g_ ) noexcept {
    g_ ^= g_ >> 16;
    g_ ^= g_ >> 8;
    g_ ^= g_ >> 4;
    g_ ^= g_ >> 2;
    g_ ^= g_ >> 1;
    return g_;
}

constexpr std::uint64_t gray2dec ( std::uint64_t g_ ) noexcept {
    g_ ^= g_ >> 32;
    g_ ^= g_ >> 16;
    g_ ^= g_ >> 8;
    g_ ^= g_ >> 4;
    g_ ^= g_ >> 2;
    g_ ^= g_ >> 1;
    return g_;
}

std::uint16_t mod_mul_inv ( const std::uint16_t a_ ) noexcept;
std::uint32_t mod_mul_inv ( const std::uint32_t a_ ) noexcept;
std::uint64_t mod_mul_inv ( const std::uint64_t a_ ) noexcept;


// Deterministic primality test.
bool is_prime ( const std::uint32_t n_ ) noexcept;
bool is_prime ( const std::uint64_t n_ ) noexcept;


// FNV1a c++11 constexpr compile time hash functions, 32 and 64 bit
// str should be a null terminated string literal, value should be left out
// e.g hash_32_fnv1a_const("example")
// code license: public domain or equivalent
// post: https://notes.underscorediscovery.com/constexpr-fnv1a/
constexpr std::uint32_t hash_32_fnv1a_const ( const char* const str, const std::uint32_t value = 0x811c9dc5 ) noexcept {
    return ( str [ 0 ] == '\0' ) ? value : hash_32_fnv1a_const ( &str [ 1 ], ( value ^ std::uint32_t ( str [ 0 ] ) ) * 0x1000193 );
}

constexpr std::uint64_t hash_64_fnv1a_const ( const char* const str, const std::uint64_t value = 0xcbf29ce484222325 ) noexcept {
    return ( str [ 0 ] == '\0' ) ? value : hash_64_fnv1a_const ( &str [ 1 ], ( value ^ std::uint64_t ( str [ 0 ] ) ) * 0x100000001b3 );
}

constexpr std::uint32_t hash_32_fnv1a_const ( const std::string & str_, const std::uint32_t value_ = 0x811c9dc5 ) noexcept {
    return hash_32_fnv1a_const ( str_.c_str ( ), value_ );
}

constexpr std::uint64_t hash_64_fnv1a_const ( const std::string & str_, const std::uint64_t value_ = 0xcbf29ce484222325 ) noexcept {
    return hash_64_fnv1a_const ( str_.c_str ( ), value_ );
}

// Integer Hashing.
constexpr std::uint32_t hash ( std::uint32_t x ) noexcept {
    x = ( ( x >> 16 ) ^ x ) * 0X45D9F3B;
    x = ( ( x >> 16 ) ^ x ) * 0X45D9F3B;
    x = ( ( x >> 16 ) ^ x );
    return x;
}

constexpr std::uint32_t unhash ( std::uint32_t x ) noexcept {
    x = ( ( x >> 16 ) ^ x ) * 0X119DE1F3;
    x = ( ( x >> 16 ) ^ x ) * 0X119DE1F3;
    x = ( ( x >> 16 ) ^ x );
    return x;
}

// 0x0CF3FD1B9997F637 0xAFC1530680179F87 - 0.0033298184
// 0xD6E8FEB86659FD93 0xCFEE444D8B59A89B - 0.0033215807

constexpr std::uint64_t hash ( std::uint64_t x ) noexcept {
    x = ( ( x >> 32 ) ^ x ) * 0xD6E8FEB86659FD93;
    x = ( ( x >> 32 ) ^ x ) * 0xD6E8FEB86659FD93;
    x = ( ( x >> 32 ) ^ x );
    return x;
}

constexpr std::uint64_t unhash ( std::uint64_t x ) noexcept {
    x = ( ( x >> 32 ) ^ x ) * 0xCFEE444D8B59A89B;
    x = ( ( x >> 32 ) ^ x ) * 0xCFEE444D8B59A89B;
    x = ( ( x >> 32 ) ^ x );
    return x;
}

template<typename T>
void hash_combine ( std::uint8_t & seed_, const T & v_ ) noexcept {
    // https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
    std::hash<T> hasher;
    seed_ ^= hasher ( v_ ) + 0x9E + ( seed_ << 6 ) + ( seed_ >> 2 );
}

template<typename T>
void hash_combine ( std::uint16_t & seed_, const T & v_ ) noexcept {
    // https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
    std::hash<T> hasher;
    seed_ ^= hasher ( v_ ) + 0x9'E37 + ( seed_ << 6 ) + ( seed_ >> 2 );
}

template<typename T>
void hash_combine ( std::uint32_t & seed_, const T & v_ ) noexcept {
    // https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
    std::hash<T> hasher;
    seed_ ^= hasher ( v_ ) + 0x9E37'79B9 + ( seed_ << 6 ) + ( seed_ >> 2 );
}

template<typename T>
void hash_combine ( std::uint64_t & seed_, const T & v_ ) noexcept {
    // https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
    std::hash<T> hasher;
    seed_ ^= hasher ( v_ ) + 0x9E3779B9'7F4A7C15 + ( seed_ << 6 ) + ( seed_ >> 2 );
}

// An invertible function used to mix the bits, borrowed from murmurhash.
constexpr std::uint64_t fmix64 ( std::uint64_t k ) noexcept {
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCD;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53;
    k ^= k >> 33;
    return k;
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr std::uint32_t popCount ( const T x_ ) noexcept {
    if constexpr ( std::is_same<T, std::uint64_t>::value ) {
        return ( std::uint32_t ) __popcnt64 ( x_ );
    }
    else {
        return ( std::uint32_t ) __popcnt ( ( std::uint32_t ) x_ );
    }
}

template < typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
constexpr T make_odd ( const T i_ ) noexcept {
    return i_ | T ( 1 );
}

template < typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
constexpr T make_even ( const T i_ ) noexcept {
    return i_ & ~T ( 1 );
}

template < typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
T bit_xor ( const T l_, const T r_ ) noexcept {
    return l_ ^ r_;
}

// Random.

// Seeding, from Intel Broadwell CPU onwards.
void seed ( std::uint8_t  & s_ ) noexcept;
void seed ( std::uint16_t & s_ ) noexcept;
void seed ( std::uint32_t & s_ ) noexcept;
void seed ( std::uint64_t & s_ ) noexcept;

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
T seed ( ) noexcept {
    T rv;
    seed ( rv );
    return rv;
}

// http://xoroshiro.di.unimi.it/

// xoroshiro128plus: This is the successor to xorshift128+. It is the
// fastest full-period generator passing BigCrush without systematic
// failures, but due to the relatively short period it is acceptable only
// for applications with a mild amount of parallelism; otherwise, use a
// xorshift1024* generator.
//
// Beside passing BigCrush, this generator passes the PractRand test suite
// up to (and included) 16TB, with the exception of binary rank tests,
// which fail due to the lowest bit being an LFSR; all other bits pass all
// tests. We suggest to use a sign test to extract a random Boolean value.
//
// The state must be seeded so that it is not everywhere zero. If you have
// a 64-bit seed, we suggest to seed a splitmix64 generator and use its
// output to fill s.
//
// Period 2 ^ 128 - 1.

struct xoroshiro128plus {

    using result_type = std::uint64_t;

    constexpr static result_type min ( ) {
        return std::numeric_limits<result_type>::min ( );
    }

    constexpr static result_type max ( ) {
        return std::numeric_limits<result_type>::max ( );
    }

    xoroshiro128plus ( ) noexcept;
    xoroshiro128plus ( const std::uint64_t s_ ) noexcept;

    void seed ( const std::uint64_t s_ ) noexcept;

    result_type operator ( ) ( ) noexcept {
        const std::uint64_t r = m_s0 + m_s1;
        m_s1 ^= m_s0;
        m_s0 = rotl ( m_s0, 24 ) ^ m_s1 ^ ( m_s1 << 16 );
        m_s1 = rotl ( m_s1, 37 );
        return r;
    }

    // This is the jump function for the generator. It is equivalent
    // to 2^64 calls to next(); it can be used to generate 2^64
    // non-overlapping subsequences for parallel computations.
    void jump ( ) noexcept;

    private:

    static inline result_type rotl ( const result_type x, int k ) noexcept {
        return ( x << k ) | ( x >> ( sizeof ( result_type ) - k ) );
    }

    std::uint64_t m_s0, m_s1;
};

// #ifdef __AVX2__

struct xoroshiro4x128plusavx {

    using result_type = std::uint64_t;

    constexpr static result_type min ( ) {
        return std::numeric_limits<result_type>::min ( );
    }

    constexpr static result_type max ( ) {
        return std::numeric_limits<result_type>::max ( );
    }

    //System seeded by default.
    xoroshiro4x128plusavx ( ) noexcept;
    xoroshiro4x128plusavx ( const std::uint64_t s_ ) noexcept;

    void seed ( const std::uint64_t s_ ) noexcept;
    result_type operator ( ) ( ) noexcept;

    private:

    __declspec ( align ( 32 ) ) __m256i m_s0, m_s1, m_r;

    static constexpr std::size_t start_case ( ) noexcept {
        return std::size_t { 3 };
    }

    std::size_t m_i;
};

// #endif

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
void print_bits ( const T n ) noexcept {
    T i = T ( 1 ) << ( sizeof ( T ) * 8 - 1 );
    while ( i ) {
        putchar ( int ( ( n & i ) > 0 ) + int ( 48 ) );
        i >>= 1;
    }
}

#ifdef __AVX2__
void print_bits ( __m256i n ) noexcept;
void print_u64 ( __m256i n ) noexcept;
#endif

}
