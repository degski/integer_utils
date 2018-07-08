
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

#ifdef _WIN32 // needed to allow binary stdout on windows
#include <fcntl.h>
#include <io.h>
#endif

#include <cstdint>
#include <iostream>

#ifndef __AVX2__
#define __AVX2__ 1
#endif

#include "../integer_utils.hpp"


using Generator = iu::xoroshiro4x128plusavx;


auto main ( ) -> int {

#ifdef _WIN32 // Needed to allow binary stdout on Windhoze...
    _setmode ( _fileno ( stdout ), _O_BINARY );
#endif

    using result_type = typename Generator::result_type;

    Generator rng ( 0xBEAC0467EBA5FACB );

    const std::size_t page_size = 524'288, buffer_size = page_size / sizeof ( result_type );
    result_type buffer [ buffer_size ];

    while ( true ) {
        for ( std::size_t i = 0; i < buffer_size; ++i ) {
            buffer [ i ] = rng ( );
        }
        std::cout.write ( reinterpret_cast<char*>( buffer ), page_size );
    }
}
