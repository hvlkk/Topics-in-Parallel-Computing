#include <stdio.h>
#include <omp.h>

#define UPTO 10000000

long int count,      /* number of primes */
         lastprime;  /* the last prime found */


void serial_primes(long int n) {
	long int i, num, divisor, quotient, remainder;

	if (n < 2) return;
	count = 1;                         /* 2 is the first prime */
	lastprime = 2;

	for (i = 0; i < (n-1)/2; ++i) {    /* For every odd number */
		num = 2*i + 3;

		divisor = 1;
		do
		{
			divisor += 2;                  /* Divide by the next odd */
			quotient  = num / divisor;
			remainder = num % divisor;
		} while (remainder && divisor <= quotient);  /* Don't go past sqrt */

		if (remainder || divisor == num) /* num is prime */
		{
			count++;
			lastprime = num;
		}
	}
}


void openmp_primes(long int n) {
	long int i, num, divisor, quotient, remainder;

	if (n < 2) return;
	count = 1;                         /* 2 is the first prime */
	lastprime = 2;

	/*
	 * Parallelize the serial algorithm but you are NOT allowed to change it!
	 * Don't add/remove/change global variables
	 */

    omp_set_dynamic(0);    // to ensure we have the exact amount of threads intended

    #pragma omp parallel for num_threads(4) schedule(guided) private(i, num, divisor, quotient, remainder) reduction(+: count) reduction(max: lastprime)
    for (i = 0; i < (n - 1) / 2; ++i) {    /* For every odd number */
        num = 2 * i + 3;

        divisor = 1;
        do
        {
            divisor += 2;                  /* Divide by the next odd */
            quotient = num / divisor;
            remainder = num % divisor;
        } while (remainder && divisor <= quotient);  /* Don't go past sqrt */


        if (remainder || divisor == num) /* num is prime */
        {
            count++;
            lastprime = num;
        }
    }
}


int main()
{
	printf("Serial and parallel prime number calculations:\n\n");

	/* Time the following to compare performance
	 */

    double currTime = omp_get_wtime();
	serial_primes(UPTO);        /* time it */
	double serialPrimesTime = omp_get_wtime();

	printf("[serial] count = %ld, last = %ld (time = %.2fsec)\n", count, lastprime, serialPrimesTime - currTime);

	serialPrimesTime = omp_get_wtime(); // updating just to get the latest time possible
	openmp_primes(UPTO);        /* time it */
	currTime = omp_get_wtime();
	printf("[openmp] count = %ld, last = %ld (time = %.2fsec)\n", count, lastprime, currTime - serialPrimesTime);


	return 0;
}
