/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
#include <x86intrin.h>
#include <emmintrin.h> /* where intrinsics are defined */
#include <stdio.h>
/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    

void transpose( int n, int blocksize, float *dst, float *src );
void transpose( int n, int blocksize, float *dst, float *src ) {
    int i,j,k,l, c, q, ir, in;
    int nblocksize = n * blocksize;
    int nbbs = n/blocksize;
    int knblocksize, kblocksize, qblocksize, rblocksize;
    /* TO DO: implement blocking (two more loops) */
    for (k = 0; k < nbbs; k++) {
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
    c = n - (n % blocksize);
    for (k = 0; k < (n % blocksize); k++) {
	for (i = 0; i < n; i++) {
	    int bananas = k + c + i * n;
	    int pajamas = i + c * n + k * n;
	    dst[bananas] = src[pajamas];
	    dst[pajamas] = src[bananas];
	}
    }
}

float d[1024] __attribute__((aligned (16)));
float Q[1024 * 1024] __attribute__((aligned (16)));
	
float ck[4] = {1.0, 1.0, 1.0, 1.0};
float dl[4] = {0.0, 0.0, 0.0, 0.0};
void square_sgemm (int n, float* A, float* B, float* C)
{
    if (n == 64) {
	__m128 b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, zero = _mm_loadu_ps(dl);
	//__m128 b1, zero=_mm_loadu_ps(dl);
	float* Aink;
	float* Bjnk;
	float* gurk;
	transpose(n, n, Q, A);
	/* For each row i of A */
	float cij;
	//int n2 = n*n;
	int jn, j, i, k;

	for (i = 0; i < n; ++i) {
	    gurk = Q + i * n;
	    b5 = _mm_loadu_ps(gurk);
	    b6 = _mm_loadu_ps(gurk + 4);
	    b7 = _mm_loadu_ps(gurk + 8);
	    b8 = _mm_loadu_ps(gurk + 12);
	    b9 = _mm_loadu_ps(gurk + 16);
	    b10 = _mm_loadu_ps(gurk + 20);
	    b11 = _mm_loadu_ps(gurk + 24);
	    b12 = _mm_loadu_ps(gurk + 28);
	    b13 = _mm_loadu_ps(gurk + 32);
	    b14 = _mm_loadu_ps(gurk + 36);
	    b15 = _mm_loadu_ps(gurk + 40);
	    b16 = _mm_loadu_ps(gurk + 44);
	    b17 = _mm_loadu_ps(gurk + 48);
	    b18 = _mm_loadu_ps(gurk + 52);
	    b19 = _mm_loadu_ps(gurk + 56);
	    b20 = _mm_loadu_ps(gurk + 60);
	    for (j = 0; j < n; j++) {
		jn = j * n;
		/* Compute C(i,j) */
		Aink = gurk;
		Bjnk = B + jn;
		b1 = _mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk + 12)))));
		b2 = _mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk + 28)))));
		b3 = _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk + 32)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+40)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk + 44)))));
		b4 = _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+56)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk + 60)))));
		for( k = 64; k < n; k += 64 ) {
		    Bjnk += 64;
		    Aink += 64;
		    b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk + 12)), b1))));
		    b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk + 28)), b2))));
		    b3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink + 32), _mm_loadu_ps(Bjnk + 32)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+36), _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+40), _mm_loadu_ps(Bjnk+40)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+44), _mm_loadu_ps(Bjnk + 44)), b3))));
		    b4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+48), _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink + 52), _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink + 56), _mm_loadu_ps(Bjnk+56)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+60), _mm_loadu_ps(Bjnk + 60)), b4))));
		}
		//b5 = _mm_hadd_ps(_mm_hadd_ps(_mm_hadd_ps(_mm_hadd_ps(b1,b2),_mm_hadd_ps(b3,b4)), zero), zero);
		b1 = _mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm_add_ps(b2, b1), _mm_add_ps(b3,b4)), zero), zero);		
		C[i + jn] += *(float*)&b1;
	    }
	}
    } else {
	__m128 one = _mm_loadu_ps(ck);
	transpose(n, n, Q, A);
	/* For each row i of A */
	float cij;
	int jn, in, j, i, k, n64 = n-64;
	float* Aink;
	float* Bjnk;
	float* gurk;
	__m128 b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12,b13, b14, b15, b16, b17, b18, b19, b20, zero = _mm_loadu_ps(dl);
	/* For each column j of B */
	for (i = 0; i < n; ++i) {
	    in = i*n;
	    gurk = Q + i * n;
	    b5 = _mm_loadu_ps(gurk);
	    b6 = _mm_loadu_ps(gurk + 4);
	    b7 = _mm_loadu_ps(gurk + 8);
	    b8 = _mm_loadu_ps(gurk + 12);
	    b9 = _mm_loadu_ps(gurk + 16);
	    b10 = _mm_loadu_ps(gurk + 20);
	    b11 = _mm_loadu_ps(gurk + 24);
	    b12 = _mm_loadu_ps(gurk + 28);
	    b13 = _mm_loadu_ps(gurk + 32);
	    b14 = _mm_loadu_ps(gurk + 36);
	    b15 = _mm_loadu_ps(gurk + 40);
	    b16 = _mm_loadu_ps(gurk + 44);
	    b17 = _mm_loadu_ps(gurk + 48);
	    b18 = _mm_loadu_ps(gurk + 52);
	    b19 = _mm_loadu_ps(gurk + 56);
	    b20 = _mm_loadu_ps(gurk + 60);
	    for (j = 0; j < n; j++) {
		jn = j * n;
		/* Compute C(i,j) */
		Aink = gurk;
		Bjnk = B + jn;
		b1 = _mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk + 12)))));
		b2 = _mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk + 28)))));
		b3 = _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk + 32)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+40)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk + 44)))));
		b4 = _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+56)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk + 60)))));
		for( k = 64; k <= n64; k += 64 ) {
		    Aink += 64;
		    Bjnk += 64;
		    b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk + 12)), b1))));
		    b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk + 28)), b2))));
		    b3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink + 32), _mm_loadu_ps(Bjnk + 32)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+36), _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+40), _mm_loadu_ps(Bjnk+40)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+44), _mm_loadu_ps(Bjnk + 44)), b3))));
		    b4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+48), _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+52), _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+56), _mm_loadu_ps(Bjnk+56)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+60), _mm_loadu_ps(Bjnk + 60)), b4))));

		}
		b1 = _mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm_add_ps(b2, b1), _mm_add_ps(b3,b4)), zero), zero);
		cij = *(float*)&b1;
		for ( ; k < n; k++) {
		    cij += Q[k + in] * B[k + jn];
		}
		C[i+jn] += cij;
	    }
	}
    }
}
