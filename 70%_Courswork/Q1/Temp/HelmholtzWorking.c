
//-------------------- COMP3001 OPENMP COURSEWORK - W1 Part1 -----------------------------
//parallelize this program using OpenMP (multithreading + vectorization)

//In Linux, compile with gcc coursework.c -o p -O2 -fopenmp -lm -fopt-info-vec-optimized

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>


void helmholtz ( int m, int n, int it_max, double alpha, double omega, double tol );
void error_check ( int m, int n, double alpha, double u[], double f[] );
void jacobi ( int m, int n, double alpha, double omega, double u[], double f[], double tol, int it_max );
double *rhs_set ( int m, int n, double alpha );
double u_exact ( double x, double y );
double uxx_exact ( double x, double y );
double uyy_exact ( double x, double y );


int main ( ) {
  double alpha = 0.25;
  int it_max = 100;
  int m = 500;
  int n = 500;
  double omega = 1.1;
  double tol = 1.0E-08;
  double wtime;


  printf ( "\n  A program which solves the 2D Helmholtz equation.\n" );


  wtime = omp_get_wtime ( );

  //---------------------routine to optimize----------------------------------
  helmholtz ( m, n, it_max, alpha, omega, tol );

  wtime = omp_get_wtime ( ) - wtime;

  printf ( "\n  Elapsed wall clock time = %f\n", wtime );

  return 0;
}















void helmholtz ( int m, int n, int it_max, double alpha, double omega, double tol ) {
  double *f;
  int i;
  int j;
  double *u;
  /*
    Initialize the data.
  */
  f = rhs_set ( m, n, alpha );

  u = (double*) _mm_malloc(m * n * sizeof(double), 64);
  if (u == NULL) { 
      printf("\nu Memory not allocated.\n");
      system("pause");
      exit(0);
  }




  #pragma omp parallel for private(i, j) shared(u, f)
  for ( j = 0; j < n; j++ )
  {
    #pragma omp simd aligned(u:64)
    for ( i = 0; i < m; i++ )
    {
      u[i+j*m] = 0.0;
    }
  }



  /*
    Solve the Helmholtz equation.
  */
  jacobi ( m, n, alpha, omega, u, f, tol, it_max );



  /*
    Determine the error.
  */
  error_check ( m, n, alpha, u, f );


  _mm_free(f);
  _mm_free(u);

  return;
}
/******************************************************************************/







void error_check ( int m, int n, double alpha, double u[], double f[] ) {
  double error_norm;
  int i;
  int j;
  double u_norm;
  double u_true;
  double u_true_norm;
  double x;
  double y;

  u_norm = 0.0;

  #pragma omp parallel for private(i, j) shared(u) reduction(+:u_norm)
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      u_norm = u_norm + u[i+j*m] * u[i+j*m];
    }
  }

  u_norm = sqrt ( u_norm );

  u_true_norm = 0.0;
  error_norm = 0.0;

  #pragma omp parallel for private(i, j, x, y, u_true) shared(u) reduction(+:error_norm, u_true_norm)
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      u_true = u_exact ( x, y );
      error_norm = error_norm + ( u[i+j*m] - u_true ) * ( u[i+j*m] - u_true );
      u_true_norm = u_true_norm + u_true * u_true;
    }
  }

  error_norm = sqrt ( error_norm );
  u_true_norm = sqrt ( u_true_norm );

  printf ( "\n" );
  printf ( "  Computed U l2 norm :       %f\n", u_norm );
  printf ( "  Computed U_EXACT l2 norm : %f\n", u_true_norm );
  printf ( "  Error l2 norm:             %f\n", error_norm );

  return;
}
/******************************************************************************/













void jacobi ( int m, int n, double alpha, double omega, double u[], double f[], 
  double tol, int it_max ) {

  double ax;
  double ay;
  double b;
  double dx;
  double dy;
  double error;
  double error_norm;
  int i;
  int it;
  int j;
  double *u_old;
  /*
    Initialize the coefficients.
  */
  dx = 2.0 / ( double ) ( m - 1 );
  dy = 2.0 / ( double ) ( n - 1 );

  ax = - 1.0 / dx / dx;
  ay = - 1.0 / dy / dy;
  b  = + 2.0 / dx / dx + 2.0 / dy / dy + alpha;

  u_old = (double*)_mm_malloc(m * n * sizeof(double), 64);
  if (u_old == NULL) {
      printf("\nu Memory not allocated.\n");
      system("pause");
      exit(0);
  }

  for ( it = 1; it <= it_max; it++ )
  {
    error_norm = 0.0;
    /*
      Copy new solution into old.
    */




    #pragma omp parallel for private(i,j) shared(u, u_old)
    for ( j = 0; j < n; j++ )                           
    {
      #pragma omp simd aligned(u, u_old: 64)
      for ( i = 0; i < m; i++ )
      {
        u_old[i+m*j] = u[i+m*j];
      }
    }

    #pragma omp parallel for private(i, j, error) shared(u, u_old, f, ax, ay, b, omega) reduction(+:error_norm)
    for ( j = 0; j < n; j++ )           
    {
      #pragma omp simd aligned(f, u_old: 64)
      for ( i = 0; i < m; i++ )
      {
        /*
          Evaluate the residual. -- Note -- loop kernel cannot be parallelized or vectorized in its current form -> do NOT have to take actions against this problem
        */
        if ( i == 0 || i == m - 1 || j == 0 || j == n - 1 )
        {
          error = u_old[i+j*m] - f[i+j*m];
        }
        else
        {
          error = ( ax * ( u_old[i-1+j*m] + u_old[i+1+j*m] ) 
            + ay * ( u_old[i+(j-1)*m] + u_old[i+(j+1)*m] ) 
            + b * u_old[i+j*m] - f[i+j*m] ) / b;
        }
        /*
          Update the solution.
        */
        u[i+j*m] = u_old[i+j*m] - omega * error;

        /*
          Accumulate the residual error.
        */
        error_norm = error_norm + error * error;
      }
    }
    /*
      Error check.
    */
    error_norm = sqrt ( error_norm ) / ( double ) ( m * n );

    //printf ( "  %d  Residual RMS %e\n", it, error_norm );

    if ( error_norm <= tol )
    {
      break;
    }

  }

  printf ( "\n" );
  printf ( "  Total number of iterations %d\n", it );


  _mm_free ( u_old );

  return;
}
/******************************************************************************/












double *rhs_set ( int m, int n, double alpha ) {
  double *f;
  double f_norm;
  int i;
  int j;
  double x;
  double y;

  f = (double*) _mm_malloc(m * n * sizeof(double), 64);
  if (f == NULL) {
      printf("\nf Memory not allocated.\n");
      system("pause");
      exit(0);
  }




  #pragma omp parallel for private(i, j) shared(f)
  for ( j = 0; j < n; j++ )
  {
    #pragma omp simd aligned(f: 64)
    for ( i = 0; i < m; i++ )
    {
      f[i+j*m] = 0.0;
    }
  }

    #pragma omp parallel for private(i, x, y) shared(f)
    for ( i = 0; i < m; i++ )
    {
      j = 0;
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      f[i+j*m] = u_exact ( x, y );
    }

    #pragma omp parallel for private(i, x, y) shared(f)
    for ( i = 0; i < m; i++ )
    {
      j = n - 1;
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      f[i+j*m] = u_exact ( x, y );
    }

    #pragma omp parallel for private(i, x, y) shared(f)
    for ( j = 0; j < n; j++ )
    {
      i = 0;
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      f[i+j*m] = u_exact ( x, y );
    }


    #pragma omp parallel for private(i, x, y) shared(f)
    for ( j = 0; j < n; j++ )
    {
      i = m - 1;
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      f[i+j*m] = u_exact ( x, y );
    }


    #pragma omp parallel for private(i, j, x, y) shared(f, alpha)
    for ( j = 1; j < n - 1; j++ )
    {
      #pragma omp simd aligned(f: 64)
      for ( i = 1; i < m - 1; i++ )
      {
        x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
        y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
        f[i+j*m] = - uxx_exact ( x, y ) - uyy_exact ( x, y ) + alpha * u_exact ( x, y );
      }
    }  
  

  f_norm = 0.0;


  #pragma omp parallel for private(i, j) shared(f) reduction(+:f_norm)
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      f_norm = f_norm + f[i+j*m] * f[i+j*m];
    }
  }

  f_norm = sqrt ( f_norm );




  printf ( "\n" );
  printf ( "  Right hand side l2 norm = %f\n", f_norm );

  return f;
}
/******************************************************************************/

double u_exact ( double x, double y ) {
  double value;

  value = ( 1.0 - x * x ) * ( 1.0 - y * y );

  return value;
}
/******************************************************************************/

double uxx_exact ( double x, double y ) {
  double value;

  value = -2.0 * ( 1.0 + y ) * ( 1.0 - y );

  return value;
}
/******************************************************************************/

double uyy_exact ( double x, double y ) {
  double value;

  value = -2.0 * ( 1.0 + x ) * ( 1.0 - x );

  return value;
}

