#ifndef __DGA_DEBUG_H__
#define __DGA_DEBUG_H__
#include <stdio.h>

#define DGA_DEBUG_PATH "./debug.log"
#define DGA_DBG 1  //open debug or not
//#define DGA_DEBUG_INFILE //output to file or not

#ifdef DGA_DEBUG_INFILE
#define DGA_DEBUG(tag,fmt...)\
do {\
    if(tag) {\
        FILE *fp = fopen(DGA_DEBUG_PATH, "a+");\
        fprintf(fp, "%s [%d]: ", __FILE__, __LINE__);\
        fprintf(fp, fmt);\
        fclose(fp);\
    }\
}while(0)

#else
#define DGA_DEBUG(tag,fmt...)\
do {\
    if (tag){\
        fprintf(stderr, "%s[%d] ", __FILE__, __LINE__);\
        fprintf(stderr, fmt);\
    }\
} while (0)

#endif

#endif
