#include "DGA_struct.h"
#include "DGA_detect.h"
#include "DGA_debug.h"
#include "DGA_struct.h"
#include "DGA_main.h"
//#define TIME_COUNT 1

dga_s* Dga_eng = NULL;

int main(int argc, char** argv)
{
	int ret = -1;
	int i = 0;
	ret = dga_init(&Dga_eng);
	if(ret < 0){
		DGA_DEBUG(DGA_DBG, "dga detect init failed\n");
		return -1;
	}
#if  TIME_COUNT
	struct timeval start, end;
	double interval;
	gettimeofday(&start, NULL);
#endif
	FILE* fp;
	if(NULL == (fp = fopen(DGA_FILE, "r"))){
		DGA_DEBUG(DGA_DBG, "fail to open data file {%s}\n",DGA_FILE);
		return -1;
	}
	char line_buf[DGA_LINE_MAX];
	memset(line_buf, '\0', DGA_LINE_MAX*sizeof(char));
	while(fgets(line_buf, DGA_LINE_MAX,  fp) != NULL){
		line_buf[strlen(line_buf)-1] = '\0';
		float prob = dga_detect_prob(Dga_eng, line_buf, strlen(line_buf));
		printf("domain:{%s}\tscore:{%f}.\n", line_buf, prob);
		memset(line_buf, '\0', DGA_LINE_MAX*sizeof(char));
		i++;
	}
#if  TIME_COUNT
	gettimeofday(&end, NULL);
	interval = 1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
	printf("running_time:%f\n\n", interval);
	printf("data_set count:%d\n", i);
	printf("time cost per domain: %.4f ms\n", interval*1.0/(i*1000));
#endif
	fclose(fp);
	dga_destory(&Dga_eng);
	return 0;
}
