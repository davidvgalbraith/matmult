//David Galbraith cs61c-ir
//Akash Jain cs61c-br

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <emmintrin.h> /* where intrinsics are defined */

void transpose( int n, int blocksize, float *dst, float *src );
void transpose( int n, int blocksize, float *dst, float *src ) {
#pragma omp parallel
    {
#pragma omp for
	/* TO DO: implement blocking (two more loops) */
	for (int k = 0; k < n/blocksize; k++) {
	    int i, j, l, q, ir, in, nblocksize = n * blocksize, nbbs = n/blocksize, knblocksize, kblocksize, qblocksize, rblocksize;
	    knblocksize = k * nblocksize;
	    kblocksize = k * blocksize;
	    for (l = 0; l < nbbs; l++) {
		qblocksize = l*blocksize + knblocksize;
		rblocksize = kblocksize + l * nblocksize;
		for( i = 0; i < blocksize; i++ ) {
		    in = i * n + qblocksize;
		    ir = i + rblocksize;
		    for( j = 0; j < blocksize/8*8; j+=8 ) {
			dst[j+in] = src[ir+j*n];
			dst[j + 1+in] = src[ir+(j+1)*n];
			dst[j + 2+in] = src[ir+(j+2)*n];
			dst[j + 3+in] = src[ir+(j+3)*n];
			dst[j + 4+in] = src[ir+(j+4)*n];
			dst[j + 5+in] = src[ir+(j+5)*n];
			dst[j + 6+in] = src[ir+(j+6)*n];
			dst[j + 7+in] = src[ir+(j+7)*n];
		    }
		    for (q = blocksize/8*8; q < blocksize; q += 1) {
			dst[q + in] = src[ir + q * n];
		    }
		}
	    }
	}
    }
#pragma omp parallel
    {
#pragma omp for
	for (int k = 0; k < (n % blocksize); k++) {
	    int pajamas, bananas, c = n - (n % blocksize), i;
	    for (i = 0; i < n; i++) {
		bananas = k + c + i * n;
		pajamas = i + c * n + k * n;
		dst[bananas] = src[pajamas];
		dst[pajamas] = src[bananas];
	    }
	}
    }
}
float Q[1048576] __attribute__((aligned (16)));
void square_sgemm (int n, float* A, float* B, float* C)
{
    transpose(n, n/16, Q, A);
    /* For each row i of A */
#pragma omp parallel
    {
#pragma omp for
	for (int i = 0; i < n; ++i) {
	    int in = i * n, jn, j, k;
	    //__m128 b1, b2, b3, b4, zero = _mm_setzero_ps();
	    __m128 b1, b2, b3, zero = _mm_setzero_ps();
	    float cij;
	    float* Aink;
	    float* Bjnk;
	    float* gurk = Q + in;
	    /* For each column j of B */
	    for (j = 0; j < n; ++j) { 
		/* Compute C(i,j) */
		jn = j * n;
		Bjnk = B + jn;
		Aink = gurk;
		k = 0;
		//b1 = b2 = b3 = b4 = _mm_setzero_ps();
		b1 = b2 = _mm_setzero_ps();
		for( ; k < (n-63); k += 64 ) {
		    //b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk + 12)), b1))));
		    //b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), b1))));
		    //b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+32), _mm_loadu_ps(Bjnk+32)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+36), _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+40), _mm_loadu_ps(Bjnk+40)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+44), _mm_loadu_ps(Bjnk+44)), b1))));
		    //b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+48), _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+52), _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+56), _mm_loadu_ps(Bjnk+56)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+60), _mm_loadu_ps(Bjnk+60)), b1))));
		    b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk + 12)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), b1))))))));
		    b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+32), _mm_loadu_ps(Bjnk+32)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+36), _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+40), _mm_loadu_ps(Bjnk+40)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+44), _mm_loadu_ps(Bjnk+44)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+48), _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+52), _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+56), _mm_loadu_ps(Bjnk+56)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+60), _mm_loadu_ps(Bjnk+60)), b2))))))));
		    Bjnk += 64;
		    Aink += 64;
		}
		b1 = _mm_add_ps(b2, b1);
		for( ; k < (n-15); k += 16 ) {
		    b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk + 12)), b1))));
		    Bjnk += 16;
		    Aink += 16;
		}
		//b1 = _mm_add_ps(_mm_add_ps(b2, b1), _mm_add_ps(b3,b4));
		for( ; k < (n-3); k+=4) {
		    b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), b1);
		    Bjnk += 4;
		    Aink += 4;
		}
		b1 = _mm_hadd_ps(_mm_hadd_ps(b1, zero), zero);
		cij = *(float*)&b1;
		for ( ; k < n; k++) {
		    cij += Q[k + in] * B[k + jn];
		}
		C[i+jn] += cij;
	    }
	}
    }
}
