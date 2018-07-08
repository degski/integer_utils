
#include <ciso646>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream.hpp> // <iostream> + nl, sp etc. defined...
#include <iterator>
#include <list>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::experimental::filesystem;

#include <autotimer.hpp>


#ifndef __AVX2__
#define __AVX2__ 1
#endif


#include <integer_utils.hpp>
#include <immintrin.h>


int main ( ) {

    iu::xoroshiro4x128plusavx gen ( 123456 );

    std::cout << gen ( ) << nl;
    std::cout << gen ( ) << nl;
    std::cout << gen ( ) << nl;
    std::cout << gen ( ) << nl;

    std::cout << gen ( ) << nl;
    std::cout << gen ( ) << nl;
    std::cout << gen ( ) << nl;
    std::cout << gen ( ) << nl;

    return 0;
}
