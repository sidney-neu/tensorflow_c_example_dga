#export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/test/projects/tensorflow_C_api_example/dga_detect/lib

LIB += 	-L../lib \
	-L. \
	-ltensorflow \
	-ltensorflow_framework \
		
INC += 	-I. \
	-I../include \
 
SRCS = 	DGA_detect.c \
	DGA_main.c \

CFLAGS = -g3 -O0 -Wall \
	-fPIC \
	-D TF_CPP_MIN_LOG_LEVEL='2' \
	-fuse-ld=gold

TARGET:=dga_detector

OBJS=$(SRCS:.c=.o)

all::$(TARGET)

$(TARGET)::$(OBJS)
	gcc ${CFLAGS} $(OBJS) $(INC) $(LIB) -Wall -o $@
%.o::%.c
	gcc $(CFLAGS) $(INC) $(LIB) -c $<
clean::
	rm -rf $(TARGET) *.o
