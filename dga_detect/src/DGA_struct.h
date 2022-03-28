#ifndef _DGA_STRUCT_H_
#define _DGA_STRUCT_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <signal.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <sched.h>

#include "tensorflow/c/c_api.h"
#include "DGA_debug.h"

#define DGA_CONF_PATH "../conf/dga.conf"
#define CONF_LINE_LENMAX 4096

#define DGA_DETECT_PROB_THRD 0.5
#define DGA_DOMAIN_LENMAX 128
#define _INPUT_DIMS 64
#define _OUTPUT_DIMS 1

typedef struct{
  TF_Buffer* graph_def;
  TF_Graph* graph;
  TF_Status* status;
  TF_ImportGraphDefOptions* graph_opts;
  unsigned input_dim;
  unsigned output_dim;
  TF_Operation* input_op;
  TF_Output input_opout;
  TF_Tensor* input;
  TF_Tensor* output;
  char TF_inlayer[1024];
  char TF_outlayer[1024];
  TF_Operation* output_op;
  TF_Output output_opout;
  TF_Tensor* output_value;
  TF_SessionOptions* sess_opts;
  TF_Session* session;
  char dga_model_path[CONF_LINE_LENMAX];
  float prob_thred;
}dga_s;
#endif
