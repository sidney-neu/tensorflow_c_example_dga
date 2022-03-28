#include "DGA_struct.h"
#include "DGA_debug.h"
#include "DGA_detect.h"

void free_buffer(void* data, size_t length) {
	if(NULL == data){
		free(data);
		data = NULL;
	}
}

void Deallocator(void* data, size_t length, void* arg) {
	//        if(NULL != data){
	//			free(data);
	//			data = NULL;
	//		}
	// *reinterpret_cast<bool*>(arg) = true;
}

int dga_read_conf(char* confpath, dga_s** dga_sT){
	FILE* fp = NULL;
	char line_buf[CONF_LINE_LENMAX] = {'\0'};
	const char s[2] = "\"";
	char* tmp_path = NULL;
		
	if(NULL == confpath){
                 DGA_DEBUG(DGA_DBG, "dga_init conf path NULL.\n");
                return -1;
	}
	if(NULL == (fp = fopen(confpath, "r"))){
		DGA_DEBUG(DGA_DBG, "Can't open conf file %s\n", confpath);
		return -1;
	}
	while(fgets(line_buf, CONF_LINE_LENMAX, fp) != NULL){
		if('#' == line_buf[0]){
			continue;
		}
		if(strstr(line_buf,"dga_model_path")){
			line_buf[strlen(line_buf)]='\0';
			tmp_path = strtok(line_buf, s);
			if(NULL == tmp_path){
				DGA_DEBUG(DGA_DBG, "Read dga_model_path NULL.");
				goto out;
			}
			tmp_path = strtok(NULL, s);
			if(NULL == tmp_path){
				DGA_DEBUG(DGA_DBG, "Read dga_model_path str NULL.");
				goto out;
			}
			memset(&((*dga_sT)->dga_model_path), '\0', CONF_LINE_LENMAX*sizeof(char));
			strncpy((*dga_sT)->dga_model_path, tmp_path, strlen(tmp_path));
		}
		else if(strstr(line_buf,"dga_prob_thred")){
			line_buf[strlen(line_buf)]='\0';
			tmp_path = strtok(line_buf, s);
			if(NULL == tmp_path){
				DGA_DEBUG(DGA_DBG, "Read dga_model_path NULL.");
				goto out;
			}
			tmp_path = strtok(NULL, s);
			if(NULL == tmp_path){
				DGA_DEBUG(DGA_DBG, "Read dga_model_path str NULL.");
				goto out;
			}
			(*dga_sT)->prob_thred = atof(tmp_path);
			if(0 >= (*dga_sT)->prob_thred || 1 <= (*dga_sT)->prob_thred){
				DGA_DEBUG(DGA_DBG, "DGA prob threshold[%f] not in (0, 1).\n", (*dga_sT)->prob_thred);
				goto out;
			}
		}
	}
	fclose(fp);
	DGA_DEBUG(DGA_DBG, "DGA read conf file[%s] sussess.\n", confpath);
	return 0;
out:
	fclose(fp);
	return -1;
}

int dga_init(dga_s** dga_sT)
{
	uint8_t intra_op_parallelism_threads = 1;
        uint8_t inter_op_parallelism_threads = 1;
        uint8_t buf[]={0x10,intra_op_parallelism_threads,0x28,inter_op_parallelism_threads};
	int64_t in_dims[] = {1, _INPUT_DIMS};
	int num_bytes_in = _INPUT_DIMS * sizeof(float);
	*dga_sT = (dga_s*)calloc(1, sizeof(dga_s));
	dga_s* dga_st = *dga_sT;
	if(NULL == dga_st){
		DGA_DEBUG(DGA_DBG, "dga_init malloc failed\n");
		goto out;
	}
	if(0 > dga_read_conf(DGA_CONF_PATH, dga_sT)){
		DGA_DEBUG(DGA_DBG, "dga read conf file[%s] failed.\n", DGA_CONF_PATH);
		goto out;
	}
	dga_st->input_dim = _INPUT_DIMS;
	dga_st->output_dim = 1;
	// Use read_file to get graph_def as TF_Buffer*
	dga_st->graph_def = read_file(dga_st->dga_model_path);
	if(NULL == dga_st->graph_def){
		goto out;
	}
	dga_st->graph = TF_NewGraph();

	// Import graph_def into graph
	dga_st->status = TF_NewStatus();
	dga_st->graph_opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(dga_st->graph, dga_st->graph_def, dga_st->graph_opts, dga_st->status);
	if (TF_GetCode(dga_st->status) != TF_OK) {
		DGA_DEBUG(DGA_DBG, "ERROR: Unable to import graph %s", TF_Message(dga_st->status));
		goto out;
	}
	else {
		DGA_DEBUG(DGA_DBG, "Successfully imported graph\n");
	}
	dga_st->input_op = TF_GraphOperationByName(dga_st->graph, "input_1");
	dga_st->input_opout.oper = dga_st->input_op;
	dga_st->input_opout.index = 0;
	dga_st->output_op = TF_GraphOperationByName(dga_st->graph, "activation_1/Sigmoid");
	dga_st->output_opout.oper = dga_st->output_op;
	dga_st->output_opout.index = 0;
	dga_st->input = TF_AllocateTensor(TF_FLOAT, in_dims, 2, num_bytes_in);
	dga_st->output = NULL;// = TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out);
	dga_st->sess_opts = TF_NewSessionOptions();
	// reverse engineered the TF_SetConfig protocol from python code like:
    	// >> config = tf.ConfigProto();config.intra_op_parallelism_threads=7;config.SerializeToString()
	TF_SetConfig(dga_st->sess_opts, buf,sizeof(buf),dga_st->status);
	if (TF_GetCode(dga_st->status) != TF_OK)
	{
		DGA_DEBUG(DGA_DBG, "ERROR: %s\n", TF_Message(dga_st->status));
		goto out;
	}
	dga_st->session = TF_NewSession(dga_st->graph, dga_st->sess_opts, dga_st->status);
	assert(TF_GetCode(dga_st->status) == TF_OK);
	DGA_DEBUG(DGA_DBG, "Init session finished...\n");
	return 0;
out:
	dga_destory(dga_sT);
	return -1;
}

float dga_asc_num(char asc)
{
	if(asc >= 'a' && asc <= 'z'){
		return asc-'a' + 1;
	}else if(asc >= 'A' && asc <= 'Z'){
		return asc-'A' + 1;
	}else if(asc >= '0' && asc <= '9'){
		return asc-'0'+27;
	}else if(asc == '-'){
		return 37;
	}else if(asc == '.'){
		return 38;
	}else if(asc == '_'){
		return 39;
	}else{
		return 0;
	}
}

int dga_url2vec(char* url, int url_len, float**vec, int vec_len)
{
	int i;
	int j = 0;
	char* tmp_str = url;
	for(i = 0; i < vec_len; i++){ (*vec)[i] = 0;}
	if(url_len < vec_len){
		i = vec_len - url_len;
		while(i < vec_len){
			(*vec)[i] = dga_asc_num(*tmp_str);
			tmp_str++;
			i++;
		}
	}else{
		i = url_len - vec_len;
		tmp_str += i;
		while(i < url_len){
			(*vec)[j] = dga_asc_num(*tmp_str);
			tmp_str++;
			i++;
			j++;
		}
	}
	/*
	for(i = 0; i < vec_len; i++){
		printf("%f\n",(*vec)[i]);
	}
`	*/
	return 0;
}

void dga_tf_deletetensor(TF_Tensor* t)
{
	TF_DeleteTensor(t);
}

int dga_detect(dga_s* dga_st, char* url, int urlen)
{
	float* out_vals = NULL;
        int tmp_label = 0;
	int tmp_dim = 0;
        float* values = (float*)calloc(dga_st->input_dim, sizeof(float));
	if(NULL == values){
		goto out;
	}
        dga_url2vec(url, urlen, &values, dga_st->input_dim);
	tmp_dim = dga_st->input_dim*sizeof(float)>TF_TensorByteSize(dga_st->input)?TF_TensorByteSize(dga_st->input):dga_st->input_dim*sizeof(float);
        memcpy(TF_TensorData(dga_st->input), values, tmp_dim);
        TF_SessionRun(dga_st->session, NULL,
                        &(dga_st->input_opout), &dga_st->input, 1,
                        &(dga_st->output_opout), &dga_st->output, 1,
                        NULL, 0, NULL, dga_st->status);
	if(TF_GetCode(dga_st->status) != TF_OK){
		DGA_DEBUG(DGA_DBG, "TF_SessionRun run err : %s.\n", TF_Message(dga_st->status));
		return -1;
	}
        out_vals = (float*)(TF_TensorData(dga_st->output));
        /*
        for (int i = 0; i < dga_st->output_dim; ++i)
        {
                std::cout << "Output values info: " << *out_vals++ << "\n";
        }
        */
        tmp_label = out_vals[0] > DGA_DETECT_PROB_THRD ? 1 : 0; 
        free(values);
        TF_DeleteTensor(dga_st->output);
        return tmp_label;
out:
	if(NULL != values){
		free(values);
		values = NULL;
	}
        TF_DeleteTensor(dga_st->output);
	dga_destory(&dga_st);
	return -1;
}


float dga_detect_prob(dga_s* dga_st, char* url, int urlen)
{
	float tmp_prob = 0;
	float* out_vals = NULL;
	int tmp_dim = 0;
	float* values = (float*)calloc(dga_st->input_dim, sizeof(float));
	if(NULL == values){
		goto out;
	}
	dga_url2vec(url, strlen(url), &values, dga_st->input_dim);
	tmp_dim = dga_st->input_dim*sizeof(float)>TF_TensorByteSize(dga_st->input)?TF_TensorByteSize(dga_st->input):dga_st->input_dim*sizeof(float);
	memcpy(TF_TensorData(dga_st->input), values, tmp_dim);
	TF_SessionRun(dga_st->session, NULL,
			&(dga_st->input_opout), &dga_st->input, 1,
			&(dga_st->output_opout), &dga_st->output, 1,
			NULL, 0, NULL, dga_st->status);
	if(TF_GetCode(dga_st->status) != TF_OK){
		DGA_DEBUG(DGA_DBG, "TF_SessionRun run err : %s.\n", TF_Message(dga_st->status));
		return -1;
	}	
	out_vals = (float*)(TF_TensorData(dga_st->output));
	tmp_prob = *out_vals;
	TF_DeleteTensor(dga_st->output);
	free(values);
	return tmp_prob;
out:
	if(NULL != values){
		free(values);
		values = NULL;
	}
	TF_DeleteTensor(dga_st->output);
	dga_destory(&dga_st);
	return -1;
}

int dga_destory(dga_s** dga_sT)
{
	dga_s* dga_st = *dga_sT;
	TF_DeleteTensor(dga_st->input);
	TF_CloseSession(dga_st->session, dga_st->status);
	TF_DeleteSession(dga_st->session, dga_st->status);
	TF_DeleteSessionOptions(dga_st->sess_opts);
	TF_DeleteImportGraphDefOptions(dga_st->graph_opts);
	TF_DeleteGraph(dga_st->graph);
	TF_DeleteStatus(dga_st->status);
	if(NULL != dga_st){
		free(dga_st);
		dga_st = NULL;
	}
	DGA_DEBUG(DGA_DBG, "dga tensorflow detector destoryed.\n");
	return 0;
}

TF_Buffer* read_file(char* file) {
	FILE *f = fopen(file, "rb");
	TF_Buffer* buf = NULL;
	if(NULL == f){
		DGA_DEBUG(DGA_DBG, "DGA model file read failed.\n");
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);  //same as rewind(f);
	void* data = malloc(fsize);
	if(NULL == data){
		DGA_DEBUG(DGA_DBG, "DGA read model file malloc failed\n");
		goto out;
	}
	fread(data, fsize, 1, f);
	buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer;
	fclose(f);
	return buf;
out:
	if(NULL != data){
		free(data);
		data = NULL;
	}
	fclose(f);
	return NULL;
}

