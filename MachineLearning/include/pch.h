// pch.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <arrayfire.h>

#include <iostream>
#include <iomanip>

#ifndef MIN
#define MIN(a,b) ((a<b)?(a):(b))
#endif //MIN

#ifndef MAX
#define MAX(a,b) ((a>b)?(a):(b))
#endif //MAX

// somewhere in af includes related to Win10 SDK, min & max are defined, blocking af::min and af::max
#undef min 
#undef max 
template <typename T>
T min(const T& a, const T& b) { return ((a<b) ? (a) : (b)); }
template <typename T>
T max(const T& a, const T& b) { return ((a>b) ? (a) : (b)); }

#if _MSC_VER < 1800
#ifndef _HUGE_ENUF
#define _HUGE_ENUF	1e+300	/* _HUGE_ENUF*_HUGE_ENUF must overflow */
#endif /* _HUGE_ENUF */

#define INFINITY	 ((float)(_HUGE_ENUF * _HUGE_ENUF))	/* causes warning C4756: overflow in constant arithmetic (by design) */

#define NAN				((float)(INFINITY * 0.0F))

template <typename T>
inline short isnan(T number) { return !(number == number); }
template <typename T>
inline short isinf(T number) { return ((number == number) && (double(number - number) != 0.0)) ? ((double(number) > 0.0) ? 1 : -1) : false; }
#endif
