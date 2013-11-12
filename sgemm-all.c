#include <stdio.h>
#include <x86intrin.h>
#include <emmintrin.h> /* where intrinsics are defined */
/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in coulombs-major format.
 * On exit, A and B maintain their input values. */    
void transpose( int n, int blocksize, float *dst, float *src );
void chew(int n, float* A, float* B, float* C);
void kronk(int n, float* A, float* B, float* C);
void n64(int n, float* A, float* B, float* C);
void wreck(float* A, int n);
void wreckk(float* B, int n);
void fdemons (int n, float* A, float* B, float* C);

void transpose( int n, int blocksize, float *dst, float *src ) {
  int i, j, k, l, c = n - (n % blocksize), q, ir, in, cn = c*n, kc, kn;
  int nblocksize = n * blocksize, nbbs = n/blocksize;
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
          dst[j+1+in] = src[ir+(j+1)*n];
          dst[j+2+in] = src[ir+(j+2)*n];
          dst[j+3+in] = src[ir+(j+3)*n];
          dst[j+4+in] = src[ir+(j+4)*n];
          dst[j+5+in] = src[ir+(j+5)*n];
          dst[j+6+in] = src[ir+(j+6)*n];
          dst[j+7+in] = src[ir+(j+7)*n];
        }
        for (q = blocksize/8*8; q < blocksize; q++) {
          dst[q + in] = src[ir + q * n];
        }
      }
    }
  }
  for (k = 0; k < (n % blocksize); k++) {
    kc = k + c;
    kn = k * n;
    for (i = 0; i < n; i++) {
      dst[kc + i * n] = src[i + cn + kn];
      dst[i + cn + kn] = src[kc + i * n];
    }
  }
}

//  float d[1024] __attribute__((aligned (16)));
float Q[1048576] __attribute__((aligned (16)));
float E[1025*1025];
float F[1025*1025];
float G[1025*1025];

void wreck(float* A, int n) {
  int q, r,s, ww = 8-(n%8) + n;
  for (q = 0; q < n; q++) {
    for (r = 0; r < n; r++) {
      E[q*ww + r] = A[q*n+r];
      r++;
      E[q*ww + r] = A[q*n+r];
      r++;
      E[q*ww + r] = A[q*n+r];
      r++;
      E[q*ww + r] = A[q*n+r];
    }
    for (s = 0; s < ww-n; s++) {
      E[q*ww + n+s] = 0.0;
    }
  }
  q = ww*n;
  while (q < ww * ww) {
    E[q] = 0.0;
    q++;
  }
}

/*Pads B with zeroes.**/
void wreckk(float* B, int n) {
  int q, r,s, qww, qn, ww = 8-(n%8) + n;
  for (q = 0; q < n; q++) {
    qww = q * ww; qn = q * n;
    for (r = 0; r < n; r++) {
      F[qww + r] = B[qn+r];
      r++;
      F[qww + r] = B[qn+r];
      r++;
      F[qww + r] = B[qn+r];
      r++;
      F[qww + r] = B[qn+r];
    }
    for (s = 0; s < ww-n; s++) {
      F[q*ww + n+s] = 0.0;
    }
  }
  q = ww*n;
  while (q < ww * ww) {
    F[q] = 0.0;
    q++;
  }
}

__m128 bee[5000];
void square_sgemm (int n, float* A, float* B, float* C) {
  if (n == 64) {
    n64(n, A, B, C);
    return;
  } else {
    if (n < 200) {
      transpose(n, 64, Q, A);
    } else {
      transpose(n, n/4, Q, A);
    }
  }
  if (n < 200 || n % 64 == 0) {
    fdemons(n, A, B, C);
    return;
  }
  if (n % 2 == 1) {
    fdemons(n, A, B, C);
    return;
    int ww = 8 - (n % 8), g, h, qqq, in, jn, offset;
    __m128 zero = _mm_setzero_ps(), b1;
    wreck(Q, n);
    wreckk(B, n);
    n += ww;
    int bs = n/2; float cij; int j, k, i;
    for (g = 0; g < 2; g++) {
      for (h = 0; h < 2; h++) {
        for (i = g * bs; i < (g+1) * bs; ++i) {
          in = i*n;
          for (qqq = 0; qqq < bs; qqq+=2) {
            bee[qqq] = _mm_loadu_ps(E + in + qqq*2);
            bee[qqq+1] = _mm_loadu_ps(E + in + qqq*2 + bs);
            qqq += 2;
            bee[qqq] = _mm_loadu_ps(E + in + qqq*2);
            bee[qqq+1] = _mm_loadu_ps(E + in + qqq*2 + bs);
            qqq += 2;
            bee[qqq] = _mm_loadu_ps(E + in + qqq*2);
            bee[qqq+1] = _mm_loadu_ps(E + in + qqq*2 + bs);
            qqq += 2;
            bee[qqq] = _mm_loadu_ps(E + in + qqq*2);
            bee[qqq+1] = _mm_loadu_ps(E + in + qqq*2 + bs);
          }
          for (j = h*bs; j < (h+1) * bs; j++) {
            b1 = zero;
            jn = j*n;
            for(k = 0; k < bs/2-4; k+=2) {
              b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(bee[k], _mm_loadu_ps(F + jn + k*2)), _mm_mul_ps(bee[k+1], _mm_loadu_ps(F + jn + k*2 + bs))), b1);
              k += 2;
              b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(bee[k], _mm_loadu_ps(F + jn + k*2)), _mm_mul_ps(bee[k+1], _mm_loadu_ps(F + jn + k*2 + bs))), b1);
            }
            while (k < bs/2) {
              b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(bee[k], _mm_loadu_ps(F + jn + k*2)), _mm_mul_ps(bee[k+1], _mm_loadu_ps(F + jn + k*2 + bs))), b1);
              k += 2;
            }
            b1 = _mm_hadd_ps(_mm_hadd_ps(b1, zero), zero);

            G[i + jn] = *(float*)&b1;

          }
        }
      }
    }
    offset = 0;
 
    for (int k = 0; k < n * n; k++) {
      if (G[k] != 0) {
        C[k - offset] += G[k];
      } else {
        offset++;
      }
    }
  } else {
    chew(n, Q, B, C);
  }
}
/**It's magic, you know. Never believe it's not so!**/
void n64 (int n, float* A, float* B, float* C)
{
  int m, p, i, j, k;
  /**SVENGALI!!!! Crazy LIKE A FOX**/
  for (m = 0; k < n / 64; k++) {
    for ( p = 0; p < n / 64; p++) {
	  /* For each row i of A */
      for ( i = 0; i < 64; ++i) {
        /* For each column j of B */
        for ( j = 0; j < 64; ++j) { 
          /* Compute C(i,j) */
          float cij = C[i+j*n];
          for( k = 0; k < 64; k++ ) {
            cij += A[i+k*n] * B[k+j*n];
          }
          C[i+j*n] += cij;
        }
      }
    }
  }
}



/**void n64(int n, float* A, float* B, float* C) {
  transpose(n, 64, Q, A);
  __m128 b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, zero = _mm_setzero_ps();
  float* Aink;
  float* Bjnk;
  float* gurk;
  float cij;
  int jn, j, i, k;
  /* For each row i of A 
  for (i = 0; i < n; ++i) {
    gurk = Q + i * n;
    b5 = _mm_loadu_ps(gurk);
    b6 = _mm_loadu_ps(gurk+4);
    b7 = _mm_loadu_ps(gurk+8);
    b8 = _mm_loadu_ps(gurk+12);
    b9 = _mm_loadu_ps(gurk+16);
    b10 = _mm_loadu_ps(gurk+20);
    b11 = _mm_loadu_ps(gurk+24);
    b12 = _mm_loadu_ps(gurk+28);
    b13 = _mm_loadu_ps(gurk+32);
    b14 = _mm_loadu_ps(gurk+36);
    b15 = _mm_loadu_ps(gurk+40);
    b16 = _mm_loadu_ps(gurk+44);
    b17 = _mm_loadu_ps(gurk+48);
    b18 = _mm_loadu_ps(gurk+52);
    b19 = _mm_loadu_ps(gurk+56);
    b20 = _mm_loadu_ps(gurk+60);
    /* For each column j of B 
    for (j = 0; j < n; j++) {
      /* Compute C(i,j) 
      jn = j * n;
      Bjnk = B + jn;
      b1 = _mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk+12)))));
      b2 = _mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk+28)))));
      b3 = _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk+32)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+40)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk+44)))));
      b4 = _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+56)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk+60)))));
      b1 = _mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm_add_ps(b2, b1), _mm_add_ps(b3,b4)), zero), zero);
      C[i + jn] += *(float*)&b1;
    }
  }
}**/

void chew(int n, float* A, float* B, float* C) {
  int bs = n/2, j, k, i, in, jn;
  __m128 b1, b2, b5, b6, b7, b8, b9, b10, b11, b12,b13, b14, b15, b16, b17, b18, b19, b20, zero = _mm_setzero_ps();
  float* Aink;
  float* Bjnk;
  float* gurk;
  float* gluk;
  float cij;
  //Get the upper-left quadrant of C
  for (i = 0; i < bs; ++i) {
    in = i*n;
    gurk = A + in;
    gluk = gurk + bs;
    b5 = _mm_loadu_ps(gurk);
    b6 = _mm_loadu_ps(gurk+4);
    b7 = _mm_loadu_ps(gurk+8);
    b8 = _mm_loadu_ps(gurk+12);
    b9 = _mm_loadu_ps(gurk+16);
    b10 = _mm_loadu_ps(gurk+20);
    b11 = _mm_loadu_ps(gurk+24);
    b12 = _mm_loadu_ps(gurk+28);
    b13 = _mm_loadu_ps(gluk);
    b14 = _mm_loadu_ps(gluk+4);
    b15 = _mm_loadu_ps(gluk+8);
    b16 = _mm_loadu_ps(gluk+12);
    b17 = _mm_loadu_ps(gluk+16);
    b18 = _mm_loadu_ps(gluk+20);
    b19 = _mm_loadu_ps(gluk+24);
    b20 = _mm_loadu_ps(gluk+28);
    for (j = 0; j < bs; j++) {
      jn = j*n;
      Bjnk = B + jn;
      b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk+12))))), _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+8+bs)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk+12+bs))))));
      b2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk+28))))), _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+16+bs)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+20+bs)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+24+bs)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk+28+bs))))));
      Aink = gurk + 32;
      Bjnk += 32;
      k = 32;
      for( ; k < (bs-31); k+= 32) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk+12)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4+bs), _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8+bs), _mm_loadu_ps(Bjnk+8+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12+bs), _mm_loadu_ps(Bjnk+12+bs)), b1))))))));
        b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16+bs), _mm_loadu_ps(Bjnk+bs+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20+bs), _mm_loadu_ps(Bjnk+bs+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24+bs), _mm_loadu_ps(Bjnk+24+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28+bs), _mm_loadu_ps(Bjnk+28+bs)), b2))))))));
        Bjnk += 32;
        Aink += 32;
      }
      b1 = _mm_add_ps(b1,b2);
      for ( ; k < (bs-3); k += 4) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), b1));
        Bjnk += 4;
        Aink += 4;
      }
      b1 = _mm_hadd_ps(_mm_hadd_ps(b1, zero), zero);
      cij = *(float*)&b1;
      for( ; k < bs; k++) {
        //Allegedly the sum of the product of the upper-left quadrants and the product of A's upper-right and B's lower-left
        cij += A[in+k]*B[jn+k] + A[in+bs+k] * B[jn+bs+k];
      }
      C[i + jn] = cij;
    }
  }
  //Get the lower-left quadrant of C
  for (i = bs; i < n/2*2; ++i) {
    in = i*n;
    gurk = A + in;
    gluk = gurk + bs;
    b5 = _mm_loadu_ps(gurk);
    b6 = _mm_loadu_ps(gurk+4);
    b7 = _mm_loadu_ps(gurk+8);
    b8 = _mm_loadu_ps(gurk+12);
    b9 = _mm_loadu_ps(gurk+16);
    b10 = _mm_loadu_ps(gurk+20);
    b11 = _mm_loadu_ps(gurk+24);
    b12 = _mm_loadu_ps(gurk+28);
    b13 = _mm_loadu_ps(gluk);
    b14 = _mm_loadu_ps(gluk+4);
    b15 = _mm_loadu_ps(gluk+8);
    b16 = _mm_loadu_ps(gluk+12);
    b17 = _mm_loadu_ps(gluk+16);
    b18 = _mm_loadu_ps(gluk+20);
    b19 = _mm_loadu_ps(gluk+24);
    b20 = _mm_loadu_ps(gluk+28);
    for (j = 0; j < bs; j++) {
      jn = j*n;
      Bjnk = B + jn;
      b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk+12))))), _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+8+bs)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk+12+bs))))));
      b2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk+28))))), _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+16+bs)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+20+bs)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+24+bs)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk+28+bs))))));
      Aink = gurk + 32;
      Bjnk += 32;
      k = 32;
      for( ; k < (bs-31); k+= 32) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk+12)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4+bs), _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8+bs), _mm_loadu_ps(Bjnk+8+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12+bs), _mm_loadu_ps(Bjnk+12+bs)), b1))))))));
        b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16+bs), _mm_loadu_ps(Bjnk+bs+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20+bs), _mm_loadu_ps(Bjnk+bs+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24+bs), _mm_loadu_ps(Bjnk+24+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28+bs), _mm_loadu_ps(Bjnk+28+bs)), b2))))))));
        Bjnk += 32;
        Aink += 32;
      }
      b1 = _mm_add_ps(b1,b2);
      for ( ; k < (bs-3); k += 4) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), b1));
        Bjnk += 4;
        Aink += 4;
      }
      b1 = _mm_hadd_ps(_mm_hadd_ps(b1, zero), zero);
      cij = *(float*)&b1;
      for( ; k < bs; k++) {
        //Allegedly the sum of the product of the upper-left quadrants and the product of A's upper-right and B's lower-left
        cij += A[in+k]*B[jn+k] + A[in+bs+k] * B[jn+bs+k];
      }
      C[i + jn] = cij;
    }
  }
  //Get the upper-right quadrant of C
  for (i = 0; i < bs; ++i) {
    in = i*n;
    gurk = A + in;
    gluk = gurk + bs;
    b5 = _mm_loadu_ps(gurk);
    b6 = _mm_loadu_ps(gurk+4);
    b7 = _mm_loadu_ps(gurk+8);
    b8 = _mm_loadu_ps(gurk+12);
    b9 = _mm_loadu_ps(gurk+16);
    b10 = _mm_loadu_ps(gurk+20);
    b11 = _mm_loadu_ps(gurk+24);
    b12 = _mm_loadu_ps(gurk+28);
    b13 = _mm_loadu_ps(gluk);
    b14 = _mm_loadu_ps(gluk+4);
    b15 = _mm_loadu_ps(gluk+8);
    b16 = _mm_loadu_ps(gluk+12);
    b17 = _mm_loadu_ps(gluk+16);
    b18 = _mm_loadu_ps(gluk+20);
    b19 = _mm_loadu_ps(gluk+24);
    b20 = _mm_loadu_ps(gluk+28);
    for (j = bs; j < n/2*2; j++) {
      jn = j*n;
      Bjnk = B + jn;
      b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk+12))))), _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+8+bs)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk+12+bs))))));
      b2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk+28))))), _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+16+bs)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+20+bs)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+24+bs)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk+28+bs))))));
      Aink = gurk + 32;
      Bjnk += 32;
      k = 32;
      for( ; k < (bs-31); k+= 32) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk+12)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4+bs), _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8+bs), _mm_loadu_ps(Bjnk+8+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12+bs), _mm_loadu_ps(Bjnk+12+bs)), b1))))))));
        b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16+bs), _mm_loadu_ps(Bjnk+bs+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20+bs), _mm_loadu_ps(Bjnk+bs+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24+bs), _mm_loadu_ps(Bjnk+24+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28+bs), _mm_loadu_ps(Bjnk+28+bs)), b2))))))));
        Bjnk += 32;
        Aink += 32;
      }
      b1 = _mm_add_ps(b1,b2);
      for ( ; k < (bs-3); k += 4) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), b1));
        Bjnk += 4;
        Aink += 4;
      }
      b1 = _mm_hadd_ps(_mm_hadd_ps(b1, zero), zero);
      cij = *(float*)&b1;
      for( ; k < bs; k++) {
        //Allegedly the sum of the product of the upper-left quadrants and the product of A's upper-right and B's lower-left
        cij += A[in+k]*B[jn+k] + A[in+bs+k] * B[jn+bs+k];
      }
      C[i + jn] = cij;
    }
  }
  //Get the lower-right quadrant of C
  for (i = bs; i < n/2*2; ++i) {
    in = i*n;
    gurk = A + in;
    gluk = gurk + bs;
    b5 = _mm_loadu_ps(gurk);
    b6 = _mm_loadu_ps(gurk+4);
    b7 = _mm_loadu_ps(gurk+8);
    b8 = _mm_loadu_ps(gurk+12);
    b9 = _mm_loadu_ps(gurk+16);
    b10 = _mm_loadu_ps(gurk+20);
    b11 = _mm_loadu_ps(gurk+24);
    b12 = _mm_loadu_ps(gurk+28);
    b13 = _mm_loadu_ps(gluk);
    b14 = _mm_loadu_ps(gluk+4);
    b15 = _mm_loadu_ps(gluk+8);
    b16 = _mm_loadu_ps(gluk+12);
    b17 = _mm_loadu_ps(gluk+16);
    b18 = _mm_loadu_ps(gluk+20);
    b19 = _mm_loadu_ps(gluk+24);
    b20 = _mm_loadu_ps(gluk+28);
    for (j = bs; j < n/2*2; j++) {
      jn = j*n;
      Bjnk = B + jn;
      b1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk+12))))), _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+8+bs)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk+12+bs))))));
      b2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk+28))))), _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+16+bs)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+20+bs)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+24+bs)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk+28+bs))))));
      Aink = gurk + 32;
      Bjnk += 32;
      k = 32;
      for( ; k < (bs-31); k+= 32) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk+12)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4+bs), _mm_loadu_ps(Bjnk+4+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8+bs), _mm_loadu_ps(Bjnk+8+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12+bs), _mm_loadu_ps(Bjnk+12+bs)), b1))))))));
        b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16+bs), _mm_loadu_ps(Bjnk+bs+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20+bs), _mm_loadu_ps(Bjnk+bs+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24+bs), _mm_loadu_ps(Bjnk+24+bs)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28+bs), _mm_loadu_ps(Bjnk+28+bs)), b2))))))));
        Bjnk += 32;
        Aink += 32;
      }
      b1 = _mm_add_ps(b1,b2);
      for ( ; k < (bs-3); k += 4) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+bs), _mm_loadu_ps(Bjnk+bs)), b1));
        Bjnk += 4;
        Aink += 4;
      }
      b1 = _mm_hadd_ps(_mm_hadd_ps(b1, zero), zero);
      cij = *(float*)&b1;
      for( ; k < bs; k++) {
        //Allegedly the sum of the product of the upper-left quadrants and the product of A's upper-right and B's lower-left
        cij += A[in+k]*B[jn+k] + A[in+bs+k] * B[jn+bs+k];
      }
      C[i + jn] = cij;
    }
  }
}
float dl[4] = {0.0, 0.0, 0.0, 0.0};


void fdemons (int n, float* A, float* B, float* C)
{
  float cij;
  int jn, in, j, i, k, newn = n-63;
  float* Aink;
  float* Bjnk;
  float* gurk;
  __m128 b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12,b13, b14, b15, b16, b17, b18, b19, b20, zero = _mm_loadu_ps(dl);
  /* For each row i of A */
  for (i = 0; i < n; ++i) {
    in = i*n;
    gurk = Q + in;
    b5 = _mm_loadu_ps(gurk);
    b6 = _mm_loadu_ps(gurk+4);
    b7 = _mm_loadu_ps(gurk+8);
    b8 = _mm_loadu_ps(gurk+12);
    b9 = _mm_loadu_ps(gurk+16);
    b10 = _mm_loadu_ps(gurk+20);
    b11 = _mm_loadu_ps(gurk+24);
    b12 = _mm_loadu_ps(gurk+28);
    b13 = _mm_loadu_ps(gurk+32);
    b14 = _mm_loadu_ps(gurk+36);
    b15 = _mm_loadu_ps(gurk+40);
    b16 = _mm_loadu_ps(gurk+44);
    b17 = _mm_loadu_ps(gurk+48);
    b18 = _mm_loadu_ps(gurk+52);
    b19 = _mm_loadu_ps(gurk+56);
    b20 = _mm_loadu_ps(gurk+60);
    /* For each column j of B */
    for (j = 0; j < n; j++) {
      /* Compute C(i,j) */
      jn = j * n;
      Bjnk = B + jn;
      b1 = _mm_add_ps(_mm_mul_ps(b5, _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(b6, _mm_loadu_ps(Bjnk + 4)), _mm_add_ps(_mm_mul_ps(b7, _mm_loadu_ps(Bjnk+8)), _mm_mul_ps(b8, _mm_loadu_ps(Bjnk+12)))));
      b2 = _mm_add_ps(_mm_mul_ps(b9, _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(b10, _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(b11, _mm_loadu_ps(Bjnk+24)), _mm_mul_ps(b12, _mm_loadu_ps(Bjnk+28)))));
      b3 = _mm_add_ps(_mm_mul_ps(b13, _mm_loadu_ps(Bjnk+32)), _mm_add_ps(_mm_mul_ps(b14, _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(b15, _mm_loadu_ps(Bjnk+40)), _mm_mul_ps(b16, _mm_loadu_ps(Bjnk+44)))));
      b4 = _mm_add_ps(_mm_mul_ps(b17, _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(b18, _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(b19, _mm_loadu_ps(Bjnk+56)), _mm_mul_ps(b20, _mm_loadu_ps(Bjnk+60)))));
      Aink = gurk + 64;
      Bjnk += 64;
      for( k = 64; k < newn; k += 64 ) {
        b1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink), _mm_loadu_ps(Bjnk)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+4), _mm_loadu_ps(Bjnk+4)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+8), _mm_loadu_ps(Bjnk+8)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+12), _mm_loadu_ps(Bjnk + 12)), b1))));
        b2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+16), _mm_loadu_ps(Bjnk+16)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+20), _mm_loadu_ps(Bjnk+20)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+24), _mm_loadu_ps(Bjnk+24)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+28), _mm_loadu_ps(Bjnk+28)), b2))));
        b3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+32), _mm_loadu_ps(Bjnk+32)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+36), _mm_loadu_ps(Bjnk+36)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+40), _mm_loadu_ps(Bjnk+40)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+44), _mm_loadu_ps(Bjnk+44)), b3))));
        b4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+48), _mm_loadu_ps(Bjnk+48)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+52), _mm_loadu_ps(Bjnk+52)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+56), _mm_loadu_ps(Bjnk+56)), _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(Aink+60), _mm_loadu_ps(Bjnk+60)), b4))));
        Bjnk += 64;
        Aink += 64;
      }
      b1 = _mm_add_ps(_mm_add_ps(b2, b1), _mm_add_ps(b3,b4));;
      for ( ; k < (n-3); k += 4) {
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
  //for (int lol = 0; lol < 100000; lol++) {
  // Q[lol] = 0.0;
  ///}
}
