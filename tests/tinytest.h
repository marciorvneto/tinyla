#ifndef TINYTEST_H
#define TINYTEST_H

#include <stdio.h>
#include <math.h>

extern int tinytest_run;
extern int tinytest_passed;
extern int tinytest_failed;
extern int tinytest_current_failed;

#define INIT_TINYTEST() \
	int tinytest_run = 0; \
	int tinytest_passed = 0; \
	int tinytest_failed = 0; \
	int tinytest_current_failed = 0;

// Defining tests: suite and name
#define TEST(suite, name) \
	void suite##_##name##_test(void)


#define EXPECT_EQ(expected, actual) \
	do { \
		if((expected) != (actual)){ \
			printf("  %s:%d: Failure\n", __FILE__, __LINE__); \
			printf("  Expected: %s (evaluates to %g)\n", #expected, (double)(expected)); \
			printf("  Got:      %s (evaluates to %g)\n", #actual, (double)(actual)); \
			tinytest_current_failed = 1; \
		} \
	} while(0)

#define EXPECT_CLOSE(expected, actual, tolerance) \
    do { \
        double _e = (double)(expected); \
        double _a = (double)(actual); \
        double _t = (double)(tolerance); \
        double _diff = fabs(_e - _a); \
        if (_diff > _t) { \
            printf("  %s:%d: Failure\n", __FILE__, __LINE__); \
            printf("    Expected: %s (evaluates to %g)\n", #expected, _e); \
            printf("    Actual:   %s (evaluates to %g)\n", #actual, _a); \
            printf("    Diff:     %g (exceeds tolerance %g)\n", _diff, _t); \
            tinytest_current_failed = 1; \
        } \
    } while (0)

#define EXPECT_TRUE(condition) \
	do { \
		if(!(condition)){ \
			printf("  %s:%d: Failure\n", __FILE__, __LINE__); \
			printf("  Expected true: %s\n", #condition); \
			tinytest_current_failed = 1; \
		} \
	} while(0)

#define RUN_TEST(suite, name) \
	do { \
		tinytest_current_failed = 0; \
		printf("[RUN] %s.%s\n", #suite, #name); \
		suite##_##name##_test(); \
		tinytest_run++;\
		if(tinytest_current_failed){ \
			tinytest_failed++; \
			printf("[FAILED] %s.%s\n", #suite, #name); \
		}else { \
			tinytest_passed++; \
			printf("[OK] %s.%s\n", #suite, #name); \
		} \
	} while(0)

#define TINYTEST_REPORT() \
    do { \
        printf("\n======================================\n"); \
        printf("[==========] %d tests run.\n", tinytest_run); \
        printf("[  PASSED  ] %d tests.\n", tinytest_passed); \
        if (tinytest_failed > 0) { \
            printf("[  FAILED  ] %d tests.\n", tinytest_failed); \
        } \
        printf("======================================\n"); \
        return tinytest_failed > 0 ? 1 : 0; \
    } while (0)

#endif
