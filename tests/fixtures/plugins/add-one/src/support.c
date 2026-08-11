#include <stddef.h>
#include <stdint.h>

extern unsigned char __heap_base;

static uintptr_t heap = 0;

static uintptr_t align_up(uintptr_t value, uintptr_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static int ensure_memory(uintptr_t required_end) {
  const uintptr_t page_size = 65536;
  uintptr_t current_end = __builtin_wasm_memory_size(0) * page_size;
  if (required_end <= current_end) {
    return 1;
  }

  uintptr_t missing = required_end - current_end;
  uintptr_t pages = (missing + page_size - 1) / page_size;
  return __builtin_wasm_memory_grow(0, pages) != (size_t)-1;
}

void *memcpy(void *destination, const void *source, size_t size) {
  unsigned char *to = destination;
  const unsigned char *from = source;
  for (size_t index = 0; index < size; index++) {
    to[index] = from[index];
  }
  return destination;
}

void *memset(void *destination, int value, size_t size) {
  unsigned char *bytes = destination;
  for (size_t index = 0; index < size; index++) {
    bytes[index] = (unsigned char)value;
  }
  return destination;
}

size_t strlen(const char *value) {
  size_t length = 0;
  while (value[length] != '\0') {
    length++;
  }
  return length;
}

void *malloc(size_t size) {
  const uintptr_t alignment = 16;
  if (heap == 0) {
    heap = (uintptr_t)&__heap_base;
  }

  uintptr_t allocation = align_up(heap + sizeof(size_t), alignment);
  uintptr_t header = allocation - sizeof(size_t);
  uintptr_t end = allocation + size;
  if (end < allocation || !ensure_memory(end)) {
    return NULL;
  }

  *((size_t *)header) = size;
  heap = end;
  return (void *)allocation;
}

void free(void *pointer) { (void)pointer; }

void *realloc(void *pointer, size_t size) {
  if (pointer == NULL) {
    return malloc(size);
  }

  size_t previous_size = ((size_t *)pointer)[-1];
  void *replacement = malloc(size);
  if (replacement == NULL) {
    return NULL;
  }
  memcpy(replacement, pointer, previous_size < size ? previous_size : size);
  return replacement;
}

__attribute__((noreturn)) void abort(void) { __builtin_trap(); }
