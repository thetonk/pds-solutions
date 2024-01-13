#ifndef CURLUTILS_H_
#define CURLUTILS_H_

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <curl/curl.h>

// Struct for byte streaming.
typedef struct
{
    char *data;
    size_t current_byte_size;
    size_t total_byte_size;
} STREAM;

// Parsing result.
typedef struct
{
    uint32_t *data;
    size_t size;
} ARRAY;

size_t write_callback(void *data, size_t size, size_t nmemb, void *destination);

size_t getWikiInfo(const char *url, CURL *curl_handle);

ARRAY getWikiPartition(const char *url, int world_rank, int world_size);
ARRAY getWikiFull(const char *url);
#endif // CURLUTILS_H_
