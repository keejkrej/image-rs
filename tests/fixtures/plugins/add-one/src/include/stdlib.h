#ifndef IMAGE_RS_ADD_ONE_FIXTURE_STDLIB_H
#define IMAGE_RS_ADD_ONE_FIXTURE_STDLIB_H

#include <stddef.h>

void *malloc(size_t size);
void *realloc(void *pointer, size_t size);
void free(void *pointer);
__attribute__((noreturn)) void abort(void);

#endif
