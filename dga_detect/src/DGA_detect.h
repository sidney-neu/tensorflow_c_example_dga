#ifndef _DGA_DETECT_H_
#define _DGA_DEETCT_H_

TF_Buffer* read_file(char* file);

void free_buffer(void* data, size_t length);

void Deallocator(void* data, size_t length, void* arg);

int dga_read_conf(char* confpath, dga_s** dga_sT);

int dga_init(dga_s** dga_sT);

float dga_asc_num(char asc);

int dga_url2vec(char* url, int url_len, float**vec, int vec_len);

void dga_tf_deletetensor(TF_Tensor* t);

int dga_detect(dga_s* dga_st, char* url, int urlen);

float dga_detect_prob(dga_s* dga_st, char* url, int urlen);

int dga_destory(dga_s** dga_sT);

#endif
